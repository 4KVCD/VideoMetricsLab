// VML Modern's drop-down and spin box arrows: a thin chevron, painted (sharp
// at any display scaling) in a colour the style sheet names.
//
// A style sheet draws a combo box's or spin box's arrow only from its
// image: rule, which Qt loads as QIcon(file name), choosing the icon engine
// by the name's suffix. This engine, built into the app as a static plugin
// for the suffix "vmlchevron", answers for names that are not files:
// "chevron-down-9a8b7c.vmlchevron" is a chevron pointing down in #9a8b7c
// (theme::chevron_icon names them). No files and no SVG module needed.
// Issue #250.
#define QT_STATICPLUGIN  // this file's plugin is built in: Q_IMPORT_PLUGIN in theme.cpp

#include <QColor>
#include <QFileInfo>
#include <QIconEngine>
#include <QIconEnginePlugin>
#include <QPainter>
#include <QPainterPath>
#include <QPen>
#include <QPixmap>
#include <QStringList>

#include <algorithm>

namespace {

class ChevronIconEngine : public QIconEngine {
public:
    explicit ChevronIconEngine(const QString& name) {
        // chevron-<up|down>-<rrggbb>
        const QStringList parts = QFileInfo(name).completeBaseName().split('-');
        up_ = parts.size() == 3 && parts[1] == QLatin1String("up");
        if (parts.size() == 3) {
            colour_ = QColor(QLatin1Char('#') + parts[2]);
        }
    }

    // The V fills the width of `rect` at 10 by 6, centred: its ends inset by
    // half the pen, so that its round caps stay inside.
    void paint(QPainter* painter, const QRect& rect, QIcon::Mode, QIcon::State) override {
        if (!colour_.isValid() || rect.isEmpty()) {
            return;
        }
        const qreal pen = 1.5;
        const qreal width = std::min<qreal>(rect.width(), rect.height() * 10.0 / 6.0) - pen;
        const qreal height = width * 0.5;  // a right angle at its point
        const QPointF centre = QRectF(rect).center();
        const qreal tip = up_ ? -height / 2 : height / 2;
        QPainterPath chevron;
        chevron.moveTo(centre.x() - width / 2, centre.y() - tip);
        chevron.lineTo(centre.x(), centre.y() + tip);
        chevron.lineTo(centre.x() + width / 2, centre.y() - tip);
        painter->save();
        painter->setRenderHint(QPainter::Antialiasing, true);
        painter->setPen(QPen(colour_, pen, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
        painter->setBrush(Qt::NoBrush);
        painter->drawPath(chevron);
        painter->restore();
    }

    // As a picture (QIcon::pixmap): the chevron on a clear background, at
    // the screen's scale. Qt's own would leave the background unset.
    QPixmap scaledPixmap(const QSize& size, QIcon::Mode mode, QIcon::State state, qreal scale) override {
        QPixmap pixmap(size * scale);
        pixmap.setDevicePixelRatio(scale);
        pixmap.fill(Qt::transparent);
        QPainter painter(&pixmap);
        paint(&painter, QRect(QPoint(), size), mode, state);
        return pixmap;
    }
    QPixmap pixmap(const QSize& size, QIcon::Mode mode, QIcon::State state) override {
        return scaledPixmap(size, mode, state, 1.0);
    }

    QIconEngine* clone() const override { return new ChevronIconEngine(*this); }
    QString key() const override { return QStringLiteral("vmlchevron"); }
    bool isNull() override { return !colour_.isValid(); }

private:
    bool up_ = false;
    QColor colour_;
};

}  // namespace

class VmlChevronIconPlugin : public QIconEnginePlugin {
    Q_OBJECT
    Q_PLUGIN_METADATA(IID QIconEngineFactoryInterface_iid FILE "chevron_icons.json")

public:
    QIconEngine* create(const QString& name) override { return new ChevronIconEngine(name); }
};

#include "chevron_icons.moc"
