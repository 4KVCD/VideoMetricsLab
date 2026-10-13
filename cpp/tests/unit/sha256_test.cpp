#include "vml/sha256.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <utility>

namespace vml {
namespace {

std::string digest_of(const std::string& message, std::size_t piece) {
    Sha256 digest;
    for (std::size_t at = 0; at < message.size(); at += piece) {
        digest.update(message.data() + at, std::min(piece, message.size() - at));
    }
    return digest.hexdigest();
}

// The digests of FIPS 180-4's examples and NIST's test messages, however a
// message is cut into pieces: around a block, and where the padding takes
// a block of its own.
TEST(Sha256, GivesTheStandardsDigests) {
    const std::pair<std::string, const char*> examples[] = {
        {"", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
        {"abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"},
        {"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
         "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"},
        {"abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrst"
         "nopqrstu",
         "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1"},
        {std::string(1'000'000, 'a'), "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"},
    };
    for (const auto& [message, expected] : examples) {
        for (const std::size_t piece : {1, 55, 64, 1000}) {
            EXPECT_EQ(digest_of(message, piece), expected) << message.size() << " bytes, in pieces of " << piece;
        }
    }
}

}  // namespace
}  // namespace vml
