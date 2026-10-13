#include "vml/frame_sync.h"

#include <cctype>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include "vml/platform/system.h"

namespace vml {

namespace {

using Whole = unsigned long long;

// Whether a * b <= limit, without forming a * b (a > 0).
bool product_at_most(Whole a, Whole b, Whole limit) { return b <= limit / a; }

}  // namespace

unsigned long long multiply_add_divide(unsigned long long a, unsigned long long b, unsigned long long add,
                                       unsigned long long c) {
    // The 128-bit product, from the four 32-bit partial products.
    const Whole a_low = a & 0xFFFFFFFFULL;
    const Whole a_high = a >> 32;
    const Whole b_low = b & 0xFFFFFFFFULL;
    const Whole b_high = b >> 32;
    const Whole low_low = a_low * b_low;
    const Whole high_low = a_high * b_low;
    const Whole low_high = a_low * b_high;
    const Whole middle = (low_low >> 32) + (high_low & 0xFFFFFFFFULL) + (low_high & 0xFFFFFFFFULL);
    Whole high = a_high * b_high + (high_low >> 32) + (low_high >> 32) + (middle >> 32);
    Whole low = (middle << 32) | (low_low & 0xFFFFFFFFULL);
    low += add;
    if (low < add) {
        ++high;  // the carry
    }
    // Long division of high:low by c, one bit at a time.
    Whole quotient = 0;
    Whole remainder = 0;
    for (int bit = 127; bit >= 0; --bit) {
        const Whole next = bit >= 64 ? (high >> (bit - 64)) & 1 : (low >> bit) & 1;
        const bool overflowed = (remainder >> 63) != 0;
        remainder = (remainder << 1) | next;
        if (overflowed || remainder >= c) {
            remainder -= c;
            if (bit < 64) {
                quotient |= Whole{1} << bit;
            } else {
                throw std::overflow_error("a number beyond 64 bits");
            }
        }
    }
    return quotient;
}

long long duration_microseconds(const std::string& duration) {
    long long whole = 0;
    long long fraction = 0;
    int fraction_digits = 0;
    bool in_fraction = false;
    for (const char c : duration) {
        if (c == '.' && !in_fraction) {
            in_fraction = true;
        } else if (std::isdigit(static_cast<unsigned char>(c)) == 0) {
            throw std::invalid_argument("not a duration: " + duration);
        } else if (!in_fraction) {
            whole = whole * 10 + (c - '0');
        } else if (fraction_digits < 6) {
            fraction = fraction * 10 + (c - '0');
            ++fraction_digits;
        }
    }
    for (; fraction_digits < 6; ++fraction_digits) {
        fraction *= 10;
    }
    return whole * 1'000'000 + fraction;
}

bool by_timestamp() {
    std::string value = platform::environment_variable(pairing_variable).value_or("");
    value.erase(0, value.find_first_not_of(" \t\r\n"));
    value.erase(value.find_last_not_of(" \t\r\n") + 1);
    for (char& c : value) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return value == "timestamp";
}

std::vector<std::string> framesync_opts() {
    if (by_timestamp()) {
        return {"shortest=1", "repeatlast=0", "ts_sync_mode=nearest"};
    }
    return {"shortest=1", "repeatlast=0"};
}

Rational limit_denominator(double value, long long max_denominator) {
    // The double's exact value as a fraction: its 53-bit mantissa over a
    // power of two, reduced.
    int exponent = 0;
    const double mantissa = std::frexp(value, &exponent);
    Whole numerator = static_cast<Whole>(std::ldexp(std::abs(mantissa), 53));
    int shift = 53 - exponent;  // value = numerator / 2^shift
    if (numerator == 0) {
        return Rational{0, 1};
    }
    while (shift > 0 && (numerator & 1) == 0) {
        numerator >>= 1;
        --shift;
    }
    if (shift < 0) {
        return Rational{static_cast<long long>(numerator << -shift), 1};
    }
    if (shift > 62) {
        throw std::domain_error("a frame rate too small to time frames by");
    }
    const Whole denominator = Whole{1} << shift;
    const Whole limit = static_cast<Whole>(max_denominator);
    if (denominator <= limit) {
        return Rational{static_cast<long long>(numerator), static_cast<long long>(denominator)};
    }
    // Python's continued-fraction search, step for step.
    Whole p0 = 0, q0 = 1, p1 = 1, q1 = 0;
    Whole n = numerator, d = denominator;
    while (true) {
        const Whole a = n / d;
        const Whole q2 = q0 + a * q1;
        if (q2 > limit) {
            break;
        }
        const Whole p2 = p0 + a * p1;
        p0 = p1;
        q0 = q1;
        p1 = p2;
        q1 = q2;
        const Whole next_d = n - a * d;
        n = d;
        d = next_d;
    }
    const Whole k = (limit - q0) / q1;
    // p1/q1 against (p0 + k*p1)/(q0 + k*q1): the first is the nearer when
    // 2 * d * (q0 + k*q1) <= the exact denominator.
    if (product_at_most(2 * d, q0 + k * q1, denominator)) {
        return Rational{static_cast<long long>(p1), static_cast<long long>(q1)};
    }
    return Rational{static_cast<long long>(p0 + k * p1), static_cast<long long>(q0 + k * q1)};
}

Rational frame_rate(const VideoInfo& info) {
    if (info.fps <= 0) {
        return Rational{25, 1};
    }
    return limit_denominator(info.fps);
}

std::vector<std::string> numbering_filters(Rational rate) {
    return {"settb=AVTB", "setpts=N*" + std::to_string(1'000'000LL * rate.denominator) + "/" +
                                std::to_string(rate.numerator)};
}

std::vector<std::string> pairing_filters(Rational rate) {
    if (by_timestamp()) {
        return {"setpts=PTS-STARTPTS"};
    }
    return numbering_filters(rate);
}

std::vector<std::string> duration_args(const std::optional<std::string>& duration_limit) {
    if (!duration_limit || duration_limit->empty()) {
        return {};
    }
    if (by_timestamp()) {
        return {"-t", *duration_limit};
    }
    return {"-enc_time_base", "filter", "-t", *duration_limit};
}

std::optional<long long> frames_within(const std::optional<std::string>& duration_limit, Rational rate) {
    if (!duration_limit || duration_limit->empty()) {
        return std::nullopt;
    }
    // Frame N is kept while N * 1000000 / rate < the limit, to the
    // microsecond: the count is limit * rate / 1000000, rounded up.
    const long long limit_us = duration_microseconds(*duration_limit);
    if (limit_us <= 0) {
        return 0;
    }
    const Whole per_second = 1'000'000ULL * static_cast<Whole>(rate.denominator);
    return static_cast<long long>(multiply_add_divide(static_cast<Whole>(limit_us),
                                                      static_cast<Whole>(rate.numerator), per_second - 1, per_second));
}

}  // namespace vml
