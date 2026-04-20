#include "demo/mjpeg_server.h"

#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace {
constexpr char kBoundary[] = "frame";

#ifdef _WIN32
using SocketHandle = SOCKET;
constexpr SocketHandle kInvalidSocket = INVALID_SOCKET;
#else
using SocketHandle = int;
constexpr SocketHandle kInvalidSocket = -1;
#endif

SocketHandle fromHandle(std::intptr_t fd) { return static_cast<SocketHandle>(fd); }

std::intptr_t toHandle(SocketHandle fd) { return static_cast<std::intptr_t>(fd); }

void closeSocket(SocketHandle fd) {
#ifdef _WIN32
    closesocket(fd);
#else
    close(fd);
#endif
}

void shutdownSocket(SocketHandle fd) {
#ifdef _WIN32
    shutdown(fd, SD_BOTH);
#else
    shutdown(fd, SHUT_RDWR);
#endif
}

std::string getClientAddress(SocketHandle fd) {
    sockaddr_in addr {};
    socklen_t len = sizeof(addr);
    if (getpeername(fd, reinterpret_cast<sockaddr*>(&addr), &len) != 0) {
        return "unknown";
    }

    char ip[INET_ADDRSTRLEN] = {};
#ifdef _WIN32
    if (InetNtopA(AF_INET, &addr.sin_addr, ip, sizeof(ip)) == nullptr) {
        return "unknown";
    }
#else
    if (inet_ntop(AF_INET, &addr.sin_addr, ip, sizeof(ip)) == nullptr) {
        return "unknown";
    }
#endif

    return std::string(ip) + ":" + std::to_string(ntohs(addr.sin_port));
}

bool sendAll(SocketHandle fd, const void* data, size_t size) {
    const char* ptr = static_cast<const char*>(data);
    while (size > 0) {
#ifdef _WIN32
        const int sent = send(fd, ptr, static_cast<int>(size), 0);
#else
        const ssize_t sent = send(fd, ptr, size, MSG_NOSIGNAL);
#endif
        if (sent <= 0) {
            return false;
        }
        ptr += sent;
        size -= static_cast<size_t>(sent);
    }
    return true;
}

bool sendString(SocketHandle fd, const std::string& data) {
    return sendAll(fd, data.data(), data.size());
}

std::string buildHttpResponse(const std::string& status, const std::string& content_type,
                              const std::string& body) {
    std::ostringstream oss;
    oss << "HTTP/1.1 " << status << "\r\n";
    oss << "Content-Type: " << content_type << "\r\n";
    oss << "Content-Length: " << body.size() << "\r\n";
    oss << "Access-Control-Allow-Origin: *\r\n";
    oss << "Cache-Control: no-cache\r\n";
    oss << "Connection: close\r\n\r\n";
    oss << body;
    return oss.str();
}

std::string getRequestPath(const std::string& request) {
    std::istringstream iss(request);
    std::string method;
    std::string path;
    std::string version;
    iss >> method >> path >> version;
    if (method.empty() || path.empty()) {
        return "/";
    }
    return path;
}
}  // namespace

MjpegServer::MjpegServer(int port, FrameProvider frame_provider, JsonProvider json_provider,
                         JsonProvider layout_provider)
    : mPort(port),
      mFrameProvider(std::move(frame_provider)),
      mJsonProvider(std::move(json_provider)),
      mLayoutProvider(std::move(layout_provider)) {}

MjpegServer::~MjpegServer() { stop(); }

bool MjpegServer::start() {
    if (mRunning) {
        return true;
    }

#ifdef _WIN32
    WSADATA wsa_data {};
    if (WSAStartup(MAKEWORD(2, 2), &wsa_data) != 0) {
        std::fprintf(stderr, "[MJPEG] WSAStartup failed\n");
        return false;
    }
#endif

    SocketHandle server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd == kInvalidSocket) {
        std::perror("socket");
#ifdef _WIN32
        WSACleanup();
#endif
        return false;
    }

    int enable = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR,
               reinterpret_cast<const char*>(&enable), sizeof(enable));

    sockaddr_in addr {};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(static_cast<uint16_t>(mPort));

    if (bind(server_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        std::perror("bind");
        closeSocket(server_fd);
#ifdef _WIN32
        WSACleanup();
#endif
        return false;
    }

    if (listen(server_fd, 16) != 0) {
        std::perror("listen");
        closeSocket(server_fd);
#ifdef _WIN32
        WSACleanup();
#endif
        return false;
    }

    mServerFd = toHandle(server_fd);
    mRunning = true;
    mAcceptThread = std::thread(&MjpegServer::acceptLoop, this);
    std::printf("[MJPEG] Listening on 0.0.0.0:%d\n", mPort);
    std::fflush(stdout);
    return true;
}

void MjpegServer::stop() {
    if (!mRunning) {
        return;
    }

    mRunning = false;
    if (mServerFd != -1) {
        const SocketHandle server_fd = fromHandle(mServerFd);
        shutdownSocket(server_fd);
        closeSocket(server_fd);
        mServerFd = -1;
    }

    if (mAcceptThread.joinable()) {
        mAcceptThread.join();
    }

#ifdef _WIN32
    WSACleanup();
#endif
}

void MjpegServer::acceptLoop() {
    const SocketHandle server_fd = fromHandle(mServerFd);

    while (mRunning) {
        const SocketHandle client_fd = accept(server_fd, nullptr, nullptr);
        if (client_fd == kInvalidSocket) {
            if (mRunning) {
                std::perror("accept");
            }
            break;
        }

        std::thread(&MjpegServer::handleClient, this, toHandle(client_fd)).detach();
    }
}

void MjpegServer::handleClient(std::intptr_t client_fd_raw) {
    const SocketHandle client_fd = fromHandle(client_fd_raw);

    std::array<char, 4096> buffer {};
#ifdef _WIN32
    const int received = recv(client_fd, buffer.data(), static_cast<int>(buffer.size() - 1), 0);
#else
    const ssize_t received = recv(client_fd, buffer.data(), buffer.size() - 1, 0);
#endif
    if (received <= 0) {
        closeSocket(client_fd);
        return;
    }

    std::string request(buffer.data(), static_cast<size_t>(received));
    std::string path = getRequestPath(request);
    const std::string client_address = getClientAddress(client_fd);

    std::printf("[MJPEG] %s -> %s\n", client_address.c_str(), path.c_str());
    std::fflush(stdout);

    if (path == "/healthz") {
        const std::string response =
            buildHttpResponse("200 OK", "text/plain; charset=utf-8", "ok\n");
        sendString(client_fd, response);
        closeSocket(client_fd);
        return;
    }

    if (path == "/detections") {
        const std::string payload = mJsonProvider ? mJsonProvider() : "{\"channels\":[]}";
        const std::string response =
            buildHttpResponse("200 OK", "application/json; charset=utf-8", payload);
        sendString(client_fd, response);
        closeSocket(client_fd);
        return;
    }

    if (path == "/layout") {
        const std::string payload =
            mLayoutProvider
                ? mLayoutProvider()
                : "{\"canvas\":{\"width\":0,\"height\":0},\"channel_count\":0,\"channels\":[]}";
        const std::string response =
            buildHttpResponse("200 OK", "application/json; charset=utf-8", payload);
        sendString(client_fd, response);
        closeSocket(client_fd);
        return;
    }

    if (path != "/stream.mjpg" && path != "/") {
        const std::string response =
            buildHttpResponse("404 Not Found", "text/plain; charset=utf-8", "not found\n");
        sendString(client_fd, response);
        closeSocket(client_fd);
        return;
    }

    std::ostringstream oss;
    oss << "HTTP/1.1 200 OK\r\n";
    oss << "Content-Type: multipart/x-mixed-replace; boundary=" << kBoundary << "\r\n";
    oss << "Access-Control-Allow-Origin: *\r\n";
    oss << "Cache-Control: no-cache\r\n";
    oss << "Pragma: no-cache\r\n";
    oss << "Connection: close\r\n\r\n";
    if (!sendString(client_fd, oss.str())) {
        closeSocket(client_fd);
        return;
    }

    std::vector<unsigned char> encoded;
    std::vector<int> encode_params = {cv::IMWRITE_JPEG_QUALITY, 80};
    while (mRunning) {
        cv::Mat frame = mFrameProvider ? mFrameProvider() : cv::Mat();
        if (frame.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            continue;
        }

        encoded.clear();
        if (!cv::imencode(".jpg", frame, encoded, encode_params)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        std::ostringstream header;
        header << "--" << kBoundary << "\r\n";
        header << "Content-Type: image/jpeg\r\n";
        header << "Content-Length: " << encoded.size() << "\r\n\r\n";
        if (!sendString(client_fd, header.str()) ||
            !sendAll(client_fd, encoded.data(), encoded.size()) ||
            !sendString(client_fd, "\r\n")) {
            break;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(33));
    }

    closeSocket(client_fd);
}
