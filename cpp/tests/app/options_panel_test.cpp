// The Options panel's Performance box as each system has it: the GPU decoder
// choice, the tooltips about where VMAF and the decoding run, and the GPUs
// found; and its Range row in seconds or frames.
#include <QCheckBox>
#include <QComboBox>
#include <QLabel>
#include <QSpinBox>
#include <QTimeEdit>

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

// A range of frames is shown in frame numbers with its last frame, the note
// on CVVDP's whole seconds beside it, and read back as it was; a range of
// seconds in times, without the note.
TEST(OptionsPanel, ARangeOfFramesIsShownAndReadInFrames) {
    OptionsPanel panel({});
    VmafOptions frames;
    frames.range_in_frames = true;
    frames.range_start = 100;
    frames.range_end = 220;  // frames 100 to 219
    panel.show_options(frames, {frames});
    EXPECT_EQ(panel.range_unit_combo->currentIndex(), 1);
    EXPECT_EQ(panel.range_first_spin->value(), 100);
    EXPECT_EQ(panel.range_last_spin->value(), 219);
    EXPECT_TRUE(panel.range_frames_note->isVisibleTo(&panel));
    EXPECT_FALSE(panel.range_start_edit->isVisibleTo(&panel));
    VmafOptions read = panel.read_options({}, true, true, true, true);
    EXPECT_EQ(read.compared_range(), frames.compared_range());
    panel.range_last_spin->setValue(-1);  // End
    read = panel.read_options({}, true, true, true, true);
    EXPECT_EQ(read.compared_range(), (ComparedRange{100, 0, true}));

    VmafOptions seconds;
    seconds.range_start = 2;
    seconds.range_end = 4;
    panel.show_options(seconds, {seconds});
    EXPECT_EQ(panel.range_unit_combo->currentIndex(), 0);
    EXPECT_FALSE(panel.range_frames_note->isVisibleTo(&panel));
    EXPECT_TRUE(panel.range_start_edit->isVisibleTo(&panel));
    EXPECT_EQ(panel.read_options({}, true, true, true, true).compared_range(), seconds.compared_range());
}

// Manual shows its crop fields with the crops entered and reads them back, an
// odd number of lines taken up to even; the other modes hide them.
TEST(OptionsPanel, ManualCropsAreShownAndReadBack) {
    OptionsPanel panel({});
    VmafOptions manual;
    manual.crop_mode = CropMode::manual;
    manual.manual_source_crop = {22, 22, 0, 0};
    panel.show_options(manual, {manual});
    EXPECT_TRUE(panel.manual_crop_box->isVisibleTo(&panel));
    EXPECT_EQ(panel.source_crop_spins[0]->value(), 22);
    VmafOptions read = panel.read_options({}, true, true, true, true);
    EXPECT_EQ(read.crop_mode, CropMode::manual);
    EXPECT_EQ(read.manual_source_crop, (CropEdges{22, 22, 0, 0}));
    EXPECT_EQ(read.manual_test_crop, CropEdges{});
    QString edited;
    QObject::connect(&panel, &OptionsPanel::field_edited, [&edited](const QString& field) { edited = field; });
    panel.test_crop_spins[2]->setValue(3);  // left
    read = panel.read_options({}, true, true, true, true);
    EXPECT_EQ(read.manual_test_crop, (CropEdges{0, 0, 4, 0}));
    // The box names itself: with several rows selected, only that edge is
    // copied to each, whatever their other crops.
    const std::optional<ManualCropBox> box = manual_crop_box_of(edited);
    ASSERT_TRUE(box.has_value()) << edited.toStdString();
    VmafOptions other;
    other.manual_test_crop = {8, 8, 0, 0};
    box->in(other) = box->in(read);
    EXPECT_EQ(other.manual_test_crop, (CropEdges{8, 8, 4, 0}));
    EXPECT_EQ(other.manual_source_crop, CropEdges{});

    panel.show_options(VmafOptions{}, {VmafOptions{}});
    EXPECT_FALSE(panel.manual_crop_box->isVisibleTo(&panel));
}

}  // namespace
}  // namespace vml::app
