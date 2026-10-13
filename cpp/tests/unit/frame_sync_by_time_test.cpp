// Pairing by timestamp, held to the Python app's (timed_pairing.json, from
// frame_sync_by_time.py, which its tests hold to FFmpeg's own pairing):
// av_rescale_q's rounding, -t in a time base, and for many patterns of
// timestamps each pair and each frame's release, in order.
#include "vml/frame_sync_by_time.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "reference.h"

namespace vml {
namespace {

using Json = nlohmann::json;

const Json& timed_reference() {
    static const Json reference = test::load_reference("timed_pairing.json");
    return reference;
}

Rational rational_from(const Json& value) { return Rational{value[0].get<long long>(), value[1].get<long long>()}; }

TEST(TimedPairing, RescalesAsAvRescaleQ) {
    const Json& rescales = timed_reference()["rescales"];
    ASSERT_FALSE(rescales.empty());
    for (const Json& item : rescales) {
        EXPECT_EQ(rescale(item[0].get<long long>(), rational_from(item[1]), rational_from(item[2])),
                  item[3].get<long long>())
            << item;
    }
    for (const Json& item : timed_reference()["durations"]) {
        EXPECT_EQ(duration_in(item[0].get<std::string>(), rational_from(item[1])), item[2].get<long long>()) << item;
    }
}

TEST(TimedPairing, PairsAndReleasesAsThePythonApp) {
    const Json& pairings = timed_reference()["pairings"];
    ASSERT_FALSE(pairings.empty());
    for (const Json& pairing : pairings) {
        const std::vector<long long> test_pts = pairing["test"].get<std::vector<long long>>();
        const std::vector<long long> source_pts = pairing["source"].get<std::vector<long long>>();
        const Rational test_base = rational_from(pairing["test_base"]);
        std::size_t test_next = 0;
        std::size_t source_next = 0;
        const auto puller = [](const std::vector<long long>& pts, std::size_t& at) {
            return [&pts, &at]() -> std::optional<TimedPairs::Frame> {
                if (at == pts.size()) {
                    return std::nullopt;
                }
                const auto frame = static_cast<int>(at);
                return TimedPairs::Frame{frame, pts[at++]};
            };
        };
        Json events = Json::array();
        TimedPairs pairs(puller(test_pts, test_next), puller(source_pts, source_next), test_base,
                         rational_from(pairing["source_base"]),
                         [&events](int frame) { events.push_back({"release test", frame}); },
                         [&events](int frame) { events.push_back({"release source", frame}); });
        const std::optional<long long> stop =
            pairing["limit"].is_null()
                ? std::nullopt
                : std::optional(duration_in(pairing["limit"].get<std::string>(), test_base));
        while (const std::optional<TimedPairs::Pair> pair = pairs.next()) {
            if (stop && pair->time >= *stop) {
                break;
            }
            events.push_back({"pair", pair->test, pair->source ? Json(*pair->source) : Json(nullptr), pair->time});
        }
        pairs.close();
        EXPECT_EQ(events, pairing["events"]) << pairing["name"];
    }
}

}  // namespace
}  // namespace vml
