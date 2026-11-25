#include "job_lines.h"

#include "widgets.h"

namespace vml::app {

void JobLines::add_label(ElidedLabel* label) { labels_.push_back(label); }

void JobLines::clear() {
    slot_.clear();
    text_.clear();
    for (ElidedLabel* label : labels_) {
        label->setVisible(false);
        label->clear_text();
    }
}

void JobLines::hide_all() {
    for (ElidedLabel* label : labels_) {
        label->setVisible(false);
    }
}

void JobLines::open(int index, const QString& text) {
    if (!text_.contains(index)) {
        text_[index] = {text, QString()};
        arrange();
    }
}

void JobLines::set(int index, const QString& text, const QString& tooltip) {
    const auto line = text_.find(index);
    if (line == text_.end()) {
        return;
    }
    line->second = {text, tooltip};
    if (const auto slot = slot_.find(index); slot != slot_.end()) {
        labels_[static_cast<std::size_t>(slot->second)]->set_text(text, tooltip);
    }
}

void JobLines::close(int index) {
    if (text_.erase(index) > 0) {
        arrange();
    }
}

std::vector<int> JobLines::indices() const {
    std::vector<int> indices;
    for (const auto& [index, line] : text_) {
        indices.push_back(index);
    }
    return indices;
}

void JobLines::arrange() {
    std::vector<int> order;
    for (const auto& [index, line] : text_) {  // in index order
        if (order.size() == labels_.size()) {
            break;
        }
        order.push_back(index);
    }
    slot_.clear();
    for (std::size_t slot = 0; slot < labels_.size(); ++slot) {
        ElidedLabel* label = labels_[slot];
        if (slot < order.size()) {
            slot_[order[slot]] = static_cast<int>(slot);
            const auto& [text, tooltip] = text_.at(order[slot]);
            label->setVisible(true);
            label->set_text(text, tooltip);
        } else {
            label->setVisible(false);
            label->clear_text();
        }
    }
}

}  // namespace vml::app
