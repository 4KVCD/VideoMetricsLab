// The Notifier's count of requests still being sent, which the window's
// close waits on (MainWindow::closeEvent): a request counts until its answer,
// and one refused before it is sent never counts.

#include <QHostAddress>
#include <QSignalSpy>
#include <QTcpServer>
#include <QTcpSocket>

#include <gtest/gtest.h>

#include "notifier.h"

namespace vml::app {
namespace {

TEST(Notifier, ARequestIsPendingUntilItIsAnswered) {
    // A server on this PC that answers every request at once.
    QTcpServer server;
    ASSERT_TRUE(server.listen(QHostAddress::LocalHost));
    QObject::connect(&server, &QTcpServer::newConnection, &server, [&server] {
        QTcpSocket* socket = server.nextPendingConnection();
        QObject::connect(socket, &QTcpSocket::readyRead, socket, [socket] {
            socket->readAll();
            socket->write("HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
            socket->disconnectFromHost();
        });
    });
    Notifier notifier;
    QSignalSpy sent(&notifier, &Notifier::sent);
    NotificationRequest request;
    request.service = "Webhook";
    request.url = "http://127.0.0.1:" + std::to_string(server.serverPort()) + "/hook";
    request.content_type = "application/json";
    request.body = "{}";
    request.plain_http = true;
    notifier.send({request});
    EXPECT_EQ(notifier.pending(), 1);
    ASSERT_TRUE(sent.wait(5000));
    EXPECT_EQ(sent.at(0).at(1).toString(), QString());  // taken
    EXPECT_EQ(notifier.pending(), 0);
    // An address it won't send to is answered at once, never pending.
    request.plain_http = false;
    notifier.send({request});
    EXPECT_EQ(sent.size(), 2);
    EXPECT_EQ(notifier.pending(), 0);
}

}  // namespace
}  // namespace vml::app
