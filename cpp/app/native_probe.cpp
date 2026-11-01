#include "native_probe.h"

#include <cstring>

#include "ffmpeg_api.h"

namespace vml::app {

const std::string& probe_decoder_whitelist() {
    // A function's static: made once, safely, by whichever of the decoder's
    // and the soundtrack's threads asks first.
    static const std::string whitelist = [] {
        std::string names;
        void* iterator = nullptr;
        while (const AVCodec* codec = av_codec_iterate(&iterator)) {
            if (av_codec_is_decoder(codec) == 0 || std::strcmp(codec->name, "vvc") == 0) {
                continue;
            }
            if (!names.empty()) {
                names += ',';
            }
            names += codec->name;
        }
        return names;
    }();
    return whitelist;
}

bool restrict_probe_decoders(AVFormatContext& format) {
    // avformat_free_context frees it, also after a failed avformat_open_input
    format.codec_whitelist = av_strdup(probe_decoder_whitelist().c_str());
    return format.codec_whitelist != nullptr;
}

}  // namespace vml::app
