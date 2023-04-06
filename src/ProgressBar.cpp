//
// Created by 17110 on 2023/3/26.
//

#include "../headers/ProgressBar.h"
#include <iostream>
#include <string>
#include <algorithm>
#include <utility>

#ifdef _WIN32

#include <Windows.h>

#else
#include <sys/ioctl.h>
#include <unistd.h>
#endif

std::string int_to_string(int a) {
    std::string buffer;
    if (a == 0) return "0";
    while (a) {
        buffer.push_back(a % 10 + '0');
        a /= 10;
    }
    std::reverse(buffer.begin(), buffer.end());
    return buffer;
}


ProgressBar::ProgressBar(int total) {
    total_iters = total;
    current_iters = 0;
#ifdef _WIN32
    CONSOLE_SCREEN_BUFFER_INFO csbi;
    int columns, rows;
    GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi);
    columns = csbi.srWindow.Right - csbi.srWindow.Left + 1;
    window_width = columns;
#else
    struct winsize w;
    ioctl(fileno(stdout), TIOCGWINSZ, &w);
    window_width = (int)(w.ws_col);
#endif
//    std::cout << "window width: " << window_width << std::endl;
}

void ProgressBar::update() {
    current_iters++;
    display();
}

void ProgressBar::update(int iters) {
    current_iters += iters;
    display();
}

void ProgressBar::display() {
    std::string output;
    std::string epoch_display;
    std::string echo_prefix;
//    time_t current_time=time(nullptr);
    if (prefix.empty()) {
        echo_prefix = prefix + "iter: ";
    } else {
        echo_prefix = prefix + " iter: ";
    }
    epoch_display += std::to_string(current_iters) + "/" + std::to_string(total_iters);
    auto progress_bar_length = window_width - echo_prefix.length() - postfix.length() - epoch_display.length() - 7;
    output += "\r[" + echo_prefix + epoch_display + "]|";
    for (auto i = 0; i < progress_bar_length; i++) {
        if (i < progress_bar_length * current_iters / total_iters) {
            output += "=";
        } else if (i == progress_bar_length * current_iters / total_iters) {
            output += ">";
        } else {
            output += " ";
        }
    }
    output += "|[" + postfix + "]\r";
    std::cout << output << std::flush;
}

void ProgressBar::display_daemon() {

}

void ProgressBar::set_prefix(std::string prefix_str) {
    this->prefix = std::move(prefix_str);
}

void ProgressBar::set_postfix(std::string postfix_str) {
    this->postfix = std::move(postfix_str);
}

void ProgressBar::close() {
    if (leave) {
        std::cout << std::endl;
        return;
    }
    this->~ProgressBar();
}

ProgressBar::ProgressBar(int total, bool leave) : ProgressBar(total) {
    this->leave = leave;
}
