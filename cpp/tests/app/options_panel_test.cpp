// The Options panel's Performance box as each system has it: the GPU decoder
// choice, the tooltips about where VMAF and the decoding run, and the GPUs
// found.
#include <QCheckBox>
#include <QComboBox>
#include <QLabel>

#include <gtest/gtest.h>

#include "options_panel.h"
#include "vml/options.h"

namespace vml::app {
namespace {

// Whether one of the panel's labels says `text`.
bool shows(const OptionsPanel& panel, const QString& text) {
    for (const QLabel* label : panel.findChildren<QLabel*>()) {
        if (label->text().contains(text)) {
            return true;
        }
    }
    return false;
}

// A Mac offers its own media engine, whichever maker a video's options name;
// Windows the makers it can decode with. The tooltips name what each uses.
TEST(OptionsPanel, TheGpuDecoderChoiceIsTheSystemsOwn) {
    OptionsPanel panel({});
    VmafOptions nvidia;
    nvidia.gpu_vendor = GpuVendor::nvidia;
    panel.show_options(nvidia, {nvidia});
    const VmafOptions read = panel.read_options({}, true, true, true, true);
#ifdef __APPLE__
    ASSERT_EQ(panel.gpu_vendor_combo->count(), 1);
    EXPECT_EQ(panel.gpu_vendor_combo->itemText(0), QStringLiteral("Apple (VideoToolbox)"));
    EXPECT_EQ(read.gpu_vendor, GpuVendor::automatic);
    for (const QString& tooltip : {panel.gpu_checkbox->toolTip(), panel.vmaf_backend_combo->toolTip()}) {
        EXPECT_FALSE(tooltip.contains("CUDA")) << tooltip.toStdString();
        EXPECT_FALSE(tooltip.contains("NVIDIA")) << tooltip.toStdString();
    }
    EXPECT_TRUE(panel.gpu_checkbox->toolTip().contains("VideoToolbox"));
    EXPECT_TRUE(shows(panel, QStringLiteral("Detected GPU(s): Apple")));
#else
    ASSERT_EQ(panel.gpu_vendor_combo->count(), 4);
    EXPECT_EQ(panel.gpu_vendor_combo->itemText(1), QStringLiteral("NVIDIA"));
    EXPECT_EQ(read.gpu_vendor, GpuVendor::nvidia);
    EXPECT_FALSE(shows(panel, QStringLiteral("Detected GPU(s)")));  // none found
#endif
}

}  // namespace
}  // namespace vml::app
