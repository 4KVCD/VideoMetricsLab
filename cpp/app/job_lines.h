// The run's progress lines under the Run button: a line of text per running
// video (more where its metrics do not fit on one), in list order. The labels are made once by the window's
// layout and hidden: lines appearing and disappearing mid-run would shift
// everything below them.
#pragma once

#include <QString>

#include <map>
#include <utility>
#include <vector>

namespace vml::app {

class WrappingLabel;

class JobLines {
public:
    // One more line to show a video on (the window's layout owns it).
    void add_label(WrappingLabel* label);

    // A run starting: no video has a line, every label is hidden and empty.
    void clear();
    // The run over: every label hidden, their text kept for the next clear.
    void hide_all();

    // Gives the video `index` a line saying `text`, unless it has one.
    void open(int index, const QString& text);
    // What the video's line says, if it has one.
    void set(int index, const QString& text, const QString& tooltip);
    // Frees the video's line, the later videos moving up.
    void close(int index);

    bool has(int index) const { return text_.contains(index); }
    // The videos with a line, in index order.
    std::vector<int> indices() const;

private:
    // The videos with a line, one label each in list order. Don't give a
    // video the first free label: a later video can then sit above an
    // earlier one, the lines change places as videos finish, and a reused
    // label keeps the previous video's tooltip.
    void arrange();

    std::vector<WrappingLabel*> labels_;
    std::map<int, std::pair<QString, QString>> text_;  // each video's text and tooltip, kept to lay the lines out again
    std::map<int, int> slot_;                          // which label shows each video
};

}  // namespace vml::app
