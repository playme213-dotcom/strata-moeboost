#pragma once

#include "moeboost_tool_parser.hpp"
#include <string>
#include <vector>
#include <map>
#include <set>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iostream>
#include <regex>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <signal.h>
#include <wait.h>
#include <pwd.h>
#include <grp.h>

namespace fs = std::filesystem;

namespace moeboost {

class WorkspaceManager {
public:
    static WorkspaceManager& instance() {
        static WorkspaceManager mgr;
        return mgr;
    }

    std::string getWorkspaceDir() const {
        return workspace_dir_;
    }

    bool setWorkspaceDir(const std::string& path) {
        std::error_code ec;
        fs::path p = fs::canonical(path, ec);
        if (ec || !fs::is_directory(p)) {
            return false;
        }
        workspace_dir_ = p.string();
        return true;
    }

    std::pair<std::string, bool> resolveSafePath(const std::string& raw_input) const {
        std::error_code ec;
        std::string clean = raw_input;
        // Trim leading/trailing whitespace
        while (!clean.empty() && (clean.front() == ' ' || clean.front() == '\n' || clean.front() == '\r' || clean.front() == '\t')) clean.erase(0, 1);
        while (!clean.empty() && (clean.back() == ' ' || clean.back() == '\n' || clean.back() == '\r' || clean.back() == '\t')) clean.pop_back();

        // Expand ~ to workspace_dir_
        if (clean.rfind("~/", 0) == 0) {
            clean = workspace_dir_ + clean.substr(1);
        } else if (clean == "~") {
            clean = workspace_dir_;
        }

        fs::path base = fs::canonical(workspace_dir_, ec);
        fs::path target;
        if (fs::path(clean).is_absolute()) {
            target = fs::weakly_canonical(clean, ec);
        } else {
            target = fs::weakly_canonical(base / clean, ec);
        }

        // Case-insensitive fallback for common directories like desktop -> Desktop
        if (!fs::exists(target, ec)) {
            fs::path parent = target.parent_path();
            std::string filename = target.filename().string();
            std::string lower_fn = filename;
            for (auto& c : lower_fn) c = std::tolower(c);

            if (fs::exists(parent, ec) && fs::is_directory(parent, ec)) {
                for (const auto& entry : fs::directory_iterator(parent, ec)) {
                    std::string entry_fn = entry.path().filename().string();
                    std::string entry_lower = entry_fn;
                    for (auto& c : entry_lower) c = std::tolower(c);
                    if (entry_lower == lower_fn) {
                        target = entry.path();
                        break;
                    }
                }
            }
        }

        std::string s_base = base.string();
        std::string s_target = target.string();

        if (s_target.rfind(s_base, 0) == 0) {
            return {s_target, true};
        }
        return {s_target, false};
    }

private:
    WorkspaceManager() {
        workspace_dir_ = "/home/local-ai";
    }
    std::string workspace_dir_;
};

class NativeToolEngine {
public:
    // =========================================================================
    // 1. FILE SYSTEM, CODING & EDITING TOOLS
    // =========================================================================

    // read_workspace_file: reads file with line numbers, start line and line count bounds
    static ToolResult readWorkspaceFile(const std::string& path_str, int line_start = 1, int line_count = 300) {
        auto [safe_path, ok] = WorkspaceManager::instance().resolveSafePath(path_str);
        if (!ok) return {"error", -1, "Access denied: path is outside active workspace."};

        std::ifstream file(safe_path);
        if (!file.is_open()) return {"error", -1, "File not found or cannot be opened: " + safe_path};

        std::string line;
        int current_line = 1;
        int read_lines = 0;
        std::ostringstream out;

        while (std::getline(file, line)) {
            if (current_line >= line_start && read_lines < line_count) {
                out << current_line << ": " << line << "\n";
                read_lines++;
            }
            current_line++;
        }
        return {"success", 0, out.str()};
    }

    // write_workspace_file: create or overwrite file safely
    static ToolResult writeWorkspaceFile(const std::string& path_str, const std::string& content) {
        auto [safe_path, ok] = WorkspaceManager::instance().resolveSafePath(path_str);
        if (!ok) return {"error", -1, "Access denied: path is outside active workspace."};

        fs::path p(safe_path);
        fs::create_directories(p.parent_path());

        std::ofstream file(safe_path, std::ios::trunc);
        if (!file.is_open()) return {"error", -1, "Failed to create/write file: " + safe_path};

        file << content;
        file.close();
        return {"success", 0, "Successfully wrote " + std::to_string(content.size()) + " bytes to " + safe_path};
    }

    // patch_workspace_file: precise search-and-replace block patcher
    static ToolResult patchWorkspaceFile(const std::string& path_str, const std::string& search_pattern, const std::string& replacement) {
        auto [safe_path, ok] = WorkspaceManager::instance().resolveSafePath(path_str);
        if (!ok) return {"error", -1, "Access denied: path is outside active workspace."};

        std::ifstream in(safe_path);
        if (!in.is_open()) return {"error", -1, "Cannot open file for patching: " + safe_path};

        std::stringstream buffer;
        buffer << in.rdbuf();
        std::string content = buffer.str();
        in.close();

        size_t pos = content.find(search_pattern);
        if (pos == std::string::npos) {
            return {"error", -1, "Exact search pattern not found in " + safe_path};
        }

        content.replace(pos, search_pattern.length(), replacement);

        std::ofstream out(safe_path, std::ios::trunc);
        out << content;
        out.close();

        return {"success", 0, "Successfully patched " + safe_path};
    }

    // edit_workspace_lines: replace a 1-based range of lines [start, end]
    static ToolResult editWorkspaceLines(const std::string& path_str, int start_line, int end_line, const std::string& replacement) {
        auto [safe_path, ok] = WorkspaceManager::instance().resolveSafePath(path_str);
        if (!ok) return {"error", -1, "Access denied: path is outside active workspace."};

        std::ifstream in(safe_path);
        if (!in.is_open()) return {"error", -1, "Cannot open file: " + safe_path};

        std::vector<std::string> lines;
        std::string line;
        while (std::getline(in, line)) {
            lines.push_back(line);
        }
        in.close();

        if (start_line < 1 || start_line > (int)lines.size() + 1 || end_line < start_line - 1) {
            return {"error", -1, "Invalid line range [" + std::to_string(start_line) + ", " + std::to_string(end_line) + "]"};
        }

        std::vector<std::string> new_lines;
        for (int i = 0; i < start_line - 1 && i < (int)lines.size(); ++i) {
            new_lines.push_back(lines[i]);
        }

        if (!replacement.empty()) {
            std::stringstream repl_stream(replacement);
            std::string r_line;
            while (std::getline(repl_stream, r_line)) {
                new_lines.push_back(r_line);
            }
        }

        for (int i = end_line; i < (int)lines.size(); ++i) {
            new_lines.push_back(lines[i]);
        }

        std::ofstream out(safe_path, std::ios::trunc);
        for (const auto& l : new_lines) {
            out << l << "\n";
        }
        out.close();

        return {"success", 0, "Successfully edited lines " + std::to_string(start_line) + "-" + std::to_string(end_line) + " in " + safe_path};
    }

    // list_workspace_files: scan directory entries
    static ToolResult listWorkspaceFiles(const std::string& rel_path = ".", int max_depth = 3) {
        auto [safe_path, ok] = WorkspaceManager::instance().resolveSafePath(rel_path);
        if (!ok) return {"error", -1, "Access denied: path is outside active workspace."};

        std::ostringstream out;
        std::error_code ec;
        for (auto const& dir_entry : fs::directory_iterator(safe_path, ec)) {
            std::string filename = dir_entry.path().filename().string();
            if (filename.rfind(".", 0) == 0) continue;

            out << (dir_entry.is_directory() ? "[DIR]  " : "[FILE] ")
                << fs::relative(dir_entry.path(), safe_path).string() << "\n";
        }
        return {"success", 0, out.str()};
    }

    // =========================================================================
    // 2. SEARCH & CODE DISCOVERY TOOLS
    // =========================================================================

    // grep_workspace: search for a pattern in text files
    static ToolResult grepWorkspace(const std::string& pattern, const std::string& rel_path = ".", const std::string& glob_filter = "") {
        auto [safe_path, ok] = WorkspaceManager::instance().resolveSafePath(rel_path);
        if (!ok) return {"error", -1, "Access denied: path is outside active workspace."};

        std::ostringstream out;
        std::error_code ec;
        int matches = 0;

        std::regex re;
        bool use_regex = true;
        try {
            re = std::regex(pattern, std::regex::icase);
        } catch (const std::regex_error&) {
            use_regex = false;
        }

        for (auto const& dir_entry : fs::recursive_directory_iterator(safe_path, ec)) {
            if (!dir_entry.is_regular_file()) continue;
            std::string filename = dir_entry.path().filename().string();
            if (filename.rfind(".", 0) == 0) continue;

            if (!glob_filter.empty() && filename.find(glob_filter) == std::string::npos) {
                continue;
            }

            std::ifstream file(dir_entry.path());
            std::string line;
            int line_num = 1;
            while (std::getline(file, line)) {
                bool found = false;
                if (use_regex) {
                    try {
                        found = std::regex_search(line, re);
                    } catch (...) {
                        found = (line.find(pattern) != std::string::npos);
                    }
                } else {
                    found = (line.find(pattern) != std::string::npos);
                }

                if (found) {
                    out << fs::relative(dir_entry.path(), safe_path).string()
                        << ":" << line_num << ": " << line << "\n";
                    matches++;
                    if (matches >= 200) break;
                }
                line_num++;
            }
            if (matches >= 200) {
                out << "\n[Matches truncated at 200 hits]\n";
                break;
            }
        }
        return {"success", 0, matches > 0 ? out.str() : "No matches found for '" + pattern + "'"};
    }

    // find_workspace_files: search for files by name/glob
    static ToolResult findWorkspaceFiles(const std::string& name_pattern, const std::string& rel_path = ".") {
        auto [safe_path, ok] = WorkspaceManager::instance().resolveSafePath(rel_path);
        if (!ok) return {"error", -1, "Access denied: path is outside active workspace."};

        std::ostringstream out;
        std::error_code ec;
        int count = 0;
        for (auto const& dir_entry : fs::recursive_directory_iterator(safe_path, ec)) {
            std::string filename = dir_entry.path().filename().string();
            if (filename.find(name_pattern) != std::string::npos) {
                out << fs::relative(dir_entry.path(), safe_path).string() << "\n";
                count++;
                if (count >= 100) break;
            }
        }
        return {"success", 0, count > 0 ? out.str() : "No files matching '" + name_pattern + "'"};
    }

    // outline_file: extract functions, classes, and structs from a source file
    static ToolResult outlineFile(const std::string& path_str) {
        auto [safe_path, ok] = WorkspaceManager::instance().resolveSafePath(path_str);
        if (!ok) return {"error", -1, "Access denied: path is outside active workspace."};

        std::ifstream file(safe_path);
        if (!file.is_open()) return {"error", -1, "File not found: " + safe_path};

        std::ostringstream out;
        std::string line;
        int line_no = 1;
        static const std::regex outline_re(
            R"(^\s*(def\s+[a-zA-Z0-9_]+|class\s+[a-zA-Z0-9_]+|struct\s+[a-zA-Z0-9_]+|(?:void|int|bool|float|double|auto|[A-Za-z0-9_:]+)\s+[a-zA-Z0-9_:]+\s*\(.*?\)))"
        );

        while (std::getline(file, line)) {
            std::smatch m;
            if (std::regex_search(line, m, outline_re)) {
                out << "Line " << line_no << ": " << m[1].str() << "\n";
            }
            line_no++;
        }
        return {"success", 0, out.str().empty() ? "No definitions found in file." : out.str()};
    }

    // find_symbol_definitions: locate class, struct, function definitions across project
    static ToolResult findSymbolDefinitions(const std::string& symbol, const std::string& rel_path = ".") {
        if (symbol.empty()) return {"error", -1, "Symbol name required."};
        std::string pat = "\\b(class|def|struct|enum|fn)\\s+" + symbol + "\\b";
        return grepWorkspace(pat, rel_path);
    }

    // =========================================================================
    // 3. SYNTAX CHECKING & DEBUGGING TOOLS
    // =========================================================================

    // check_syntax: validates syntax for Python, C/C++, Shell, and JSON
    static ToolResult checkSyntax(const std::string& path_str) {
        auto [safe_path, ok] = WorkspaceManager::instance().resolveSafePath(path_str);
        if (!ok) return {"error", -1, "Access denied: path is outside active workspace."};

        std::string ext = fs::path(safe_path).extension().string();
        if (ext == ".py") {
            return executeTerminal("python3 -m py_compile \"" + safe_path + "\"");
        } else if (ext == ".cpp" || ext == ".hpp" || ext == ".cc" || ext == ".c") {
            return executeTerminal("g++ -fsyntax-only -std=c++17 \"" + safe_path + "\"");
        } else if (ext == ".sh" || ext == ".bash") {
            return executeTerminal("bash -n \"" + safe_path + "\"");
        } else if (ext == ".json") {
            return executeTerminal("python3 -m json.tool \"" + safe_path + "\" > /dev/null");
        }
        return {"success", 0, "No syntax compiler configured for " + ext + ", file exists."};
    }

    // diagnose_tool_error: automated diagnostic suggestions for failed commands or missing files
    static ToolResult diagnoseToolError(const std::string& tool_name, const std::string& target, const std::string& error_output) {
        std::ostringstream out;
        out << "Diagnostics for tool '" << tool_name << "' on target '" << target << "':\n";

        std::string err_lower = error_output;
        std::transform(err_lower.begin(), err_lower.end(), err_lower.begin(), [](unsigned char c){ return std::tolower(c); });

        if (err_lower.find("not found") != std::string::npos || err_lower.find("no such file") != std::string::npos) {
            out << "1. Path Resolution: File or command not found. Verify with list_files or find_files.\n";
            out << "2. Ensure the path is relative to the active project root.\n";
        }
        if (err_lower.find("syntaxerror") != std::string::npos || err_lower.find("expected") != std::string::npos) {
            out << "1. Syntax Error: Inspect the file with read_file and run check_syntax.\n";
        }
        if (err_lower.find("permission denied") != std::string::npos) {
            out << "1. Permission Check: Run permissions tool to check chmod/chown mode.\n";
        }
        if (err_lower.find("modulenotfounderror") != std::string::npos || err_lower.find("import error") != std::string::npos) {
            out << "1. Dependency Missing: Install the required package into the active environment.\n";
        }
        return {"success", 0, out.str()};
    }

    // run_project_checks: runs project-wide non-executing validation (syntax, JSON)
    static ToolResult runProjectChecks(const std::string& rel_path = ".") {
        auto [safe_path, ok] = WorkspaceManager::instance().resolveSafePath(rel_path);
        if (!ok) return {"error", -1, "Access denied: path is outside active workspace."};

        std::ostringstream out;
        std::error_code ec;
        int checked = 0, failed = 0;

        for (auto const& dir_entry : fs::recursive_directory_iterator(safe_path, ec)) {
            if (!dir_entry.is_regular_file()) continue;
            std::string ext = dir_entry.path().extension().string();
            if (ext == ".py" || ext == ".json" || ext == ".sh") {
                ToolResult r = checkSyntax(dir_entry.path().string());
                checked++;
                if (r.status != "success") {
                    failed++;
                    out << "[FAIL] " << fs::relative(dir_entry.path(), safe_path).string() << ": " << r.output << "\n";
                }
            }
        }
        out << "Checked " << checked << " files: " << (checked - failed) << " passed, " << failed << " failed.\n";
        return {failed == 0 ? "success" : "error", failed, out.str()};
    }

    // =========================================================================
    // 4. TERMINAL & EXECUTION TOOLS
    // =========================================================================

    // execute_terminal: run shell command with pipe, timeout, and buffer bounding
    static ToolResult executeTerminal(const std::string& command, int timeout_sec = 60, const std::string& cwd = "") {
        if (command.empty()) return {"error", -1, "Command cannot be empty."};

        std::string target_cwd = cwd.empty() ? WorkspaceManager::instance().getWorkspaceDir() : cwd;
        auto [safe_cwd, ok] = WorkspaceManager::instance().resolveSafePath(target_cwd);
        if (!ok) return {"error", -1, "Invalid working directory outside workspace."};

        int pipefd[2];
        if (pipe(pipefd) == -1) return {"error", -1, "Failed to create pipe."};

        pid_t pid = fork();
        if (pid == -1) {
            close(pipefd[0]);
            close(pipefd[1]);
            return {"error", -1, "Fork failed."};
        }

        if (pid == 0) {
            close(pipefd[0]);
            dup2(pipefd[1], STDOUT_FILENO);
            dup2(pipefd[1], STDERR_FILENO);
            close(pipefd[1]);

            if (chdir(safe_cwd.c_str()) != 0) {
                // ignore
            }
            execl("/bin/bash", "bash", "-c", command.c_str(), (char*)NULL);
            _exit(127);
        }

        close(pipefd[1]);
        std::string output;
        char buffer[4096];
        ssize_t bytes_read;

        while ((bytes_read = read(pipefd[0], buffer, sizeof(buffer) - 1)) > 0) {
            buffer[bytes_read] = '\0';
            output += buffer;
            if (output.size() > 64000) break;
        }
        close(pipefd[0]);

        int status = 0;
        waitpid(pid, &status, 0);
        int exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;

        return {exit_code == 0 ? "success" : "error", exit_code, output};
    }

    // git_status
    static ToolResult gitStatus(const std::string& rel_path = ".") {
        return executeTerminal("git status --short", 30, rel_path);
    }

    // git_diff
    static ToolResult gitDiff(const std::string& args = "", const std::string& rel_path = ".") {
        std::string cmd = "git diff " + args;
        return executeTerminal(cmd, 30, rel_path);
    }

    // =========================================================================
    // 5. DIAGNOSTICS & SYSTEM TELEMETRY
    // =========================================================================

    // check_port_open
    static ToolResult checkPortOpen(const std::string& host, int port, int timeout_ms = 1000) {
        int sock = socket(AF_INET, SOCK_STREAM, 0);
        if (sock < 0) return {"error", -1, "Socket creation failed"};

        struct sockaddr_in serv_addr;
        std::memset(&serv_addr, 0, sizeof(serv_addr));
        serv_addr.sin_family = AF_INET;
        serv_addr.sin_port = htons(port);

        if (inet_pton(AF_INET, host.c_str(), &serv_addr.sin_addr) <= 0) {
            close(sock);
            return {"error", -1, "Invalid host address"};
        }

        struct timeval tv;
        tv.tv_sec = timeout_ms / 1000;
        tv.tv_usec = (timeout_ms % 1000) * 1000;
        setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv, sizeof tv);
        setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, (const char*)&tv, sizeof tv);

        int res = connect(sock, (struct sockaddr*)&serv_addr, sizeof(serv_addr));
        close(sock);

        if (res == 0) {
            return {"success", 0, "Port " + std::to_string(port) + " on " + host + " is OPEN and accepting connections."};
        }
        return {"error", 1, "Port " + std::to_string(port) + " on " + host + " is CLOSED or not reachable."};
    }

    // inspect_workspace_permissions
    static ToolResult inspectWorkspacePermissions(const std::string& path_str) {
        auto [safe_path, ok] = WorkspaceManager::instance().resolveSafePath(path_str);
        if (!ok) return {"error", -1, "Access denied: path is outside active workspace."};

        struct stat st;
        if (stat(safe_path.c_str(), &st) != 0) {
            return {"error", -1, "Cannot stat file: " + safe_path};
        }

        struct passwd* pw = getpwuid(st.st_uid);
        struct group* gr = getgrgid(st.st_gid);

        std::ostringstream out;
        out << "Path: " << safe_path << "\n"
            << "Owner: " << (pw ? pw->pw_name : std::to_string(st.st_uid)) << "\n"
            << "Group: " << (gr ? gr->gr_name : std::to_string(st.st_gid)) << "\n"
            << "Mode: " << (st.st_mode & 0777) << "\n"
            << "Is Directory: " << (S_ISDIR(st.st_mode) ? "true" : "false") << "\n";

        return {"success", 0, out.str()};
    }

    // get_system_telemetry
    static ToolResult getSystemTelemetry() {
        std::ostringstream out;
        std::ifstream meminfo("/proc/meminfo");
        std::string line;
        long total_kb = 0, avail_kb = 0;
        while (std::getline(meminfo, line)) {
            if (line.rfind("MemTotal:", 0) == 0) {
                sscanf(line.c_str(), "MemTotal: %ld kB", &total_kb);
            } else if (line.rfind("MemAvailable:", 0) == 0) {
                sscanf(line.c_str(), "MemAvailable: %ld kB", &avail_kb);
            }
        }
        out << "RAM Total: " << (total_kb / 1024) << " MB\n";
        out << "RAM Available: " << (avail_kb / 1024) << " MB\n";
        return {"success", 0, out.str()};
    }
};

class ToolDispatcher {
public:
    static ToolResult dispatch(const std::string& raw_name, const std::map<std::string, std::string>& params) {
        std::string name = canonicalize(raw_name);

        // 1. Coding & File Editing Tools
        if (name == "read_workspace_file") {
            std::string path = getParam(params, "path", getParam(params, "file", ""));
            int start = std::stoi(getParam(params, "line_start", getParam(params, "start_line", "1")));
            int count = std::stoi(getParam(params, "line_count", "300"));
            return NativeToolEngine::readWorkspaceFile(path, start, count);
        }
        else if (name == "write_workspace_file") {
            std::string path = getParam(params, "path", "");
            std::string content = getParam(params, "content", "");
            return NativeToolEngine::writeWorkspaceFile(path, content);
        }
        else if (name == "patch_workspace_file") {
            std::string path = getParam(params, "path", "");
            std::string search = getParam(params, "search_pattern", "");
            std::string repl = getParam(params, "replacement", "");
            return NativeToolEngine::patchWorkspaceFile(path, search, repl);
        }
        else if (name == "edit_workspace_lines") {
            std::string path = getParam(params, "path", "");
            int start = std::stoi(getParam(params, "start_line", "1"));
            int end = std::stoi(getParam(params, "end_line", "1"));
            std::string repl = getParam(params, "replacement", "");
            return NativeToolEngine::editWorkspaceLines(path, start, end, repl);
        }
        else if (name == "list_workspace_files") {
            std::string path = getParam(params, "path", "");
            if (path.empty()) path = getParam(params, "relative_path", ".");
            int max_depth = std::stoi(getParam(params, "max_depth", "3"));
            return NativeToolEngine::listWorkspaceFiles(path, max_depth);
        }

        // 2. Code Search & Discovery Tools
        else if (name == "grep_workspace") {
            std::string pattern = getParam(params, "pattern", "");
            std::string path = getParam(params, "path", ".");
            std::string glob = getParam(params, "glob_filter", "");
            return NativeToolEngine::grepWorkspace(pattern, path, glob);
        }
        else if (name == "find_workspace_files") {
            std::string name_pat = getParam(params, "name_pattern", getParam(params, "pattern", ""));
            std::string path = getParam(params, "path", ".");
            return NativeToolEngine::findWorkspaceFiles(name_pat, path);
        }
        else if (name == "outline_python_file") {
            std::string path = getParam(params, "path", getParam(params, "file", ""));
            return NativeToolEngine::outlineFile(path);
        }
        else if (name == "find_symbol_definitions") {
            std::string sym = getParam(params, "symbol", getParam(params, "symbol_name", ""));
            std::string path = getParam(params, "path", ".");
            return NativeToolEngine::findSymbolDefinitions(sym, path);
        }

        // 3. Syntax Checking & Debugging Tools
        else if (name == "check_python_file") {
            return NativeToolEngine::checkSyntax(getParam(params, "path", ""));
        }
        else if (name == "diagnose_tool_error") {
            std::string tool = getParam(params, "tool_name", getParam(params, "name", ""));
            std::string target = getParam(params, "target", "");
            std::string out = getParam(params, "output", getParam(params, "error", ""));
            return NativeToolEngine::diagnoseToolError(tool, target, out);
        }
        else if (name == "run_project_checks") {
            return NativeToolEngine::runProjectChecks(getParam(params, "path", "."));
        }

        // 4. Execution & Version Control Tools
        else if (name == "execute_terminal") {
            std::string cmd = getParam(params, "command", "");
            std::string cwd = getParam(params, "cwd", "");
            int timeout = std::stoi(getParam(params, "timeout", "60"));
            return NativeToolEngine::executeTerminal(cmd, timeout, cwd);
        }
        else if (name == "git_status") {
            return NativeToolEngine::gitStatus(getParam(params, "path", "."));
        }
        else if (name == "git_diff") {
            return NativeToolEngine::gitDiff(getParam(params, "args", ""), getParam(params, "path", "."));
        }

        // 5. Diagnostics
        else if (name == "check_port_open") {
            std::string host = getParam(params, "host", "127.0.0.1");
            int port = std::stoi(getParam(params, "port", "80"));
            return NativeToolEngine::checkPortOpen(host, port);
        }
        else if (name == "inspect_workspace_permissions") {
            return NativeToolEngine::inspectWorkspacePermissions(getParam(params, "path", "."));
        }
        else if (name == "get_system_telemetry") {
            return NativeToolEngine::getSystemTelemetry();
        }

        return {"error", -1, "Unknown tool: " + raw_name};
    }

private:
    static std::string getParam(const std::map<std::string, std::string>& m, const std::string& k, const std::string& def) {
        auto it = m.find(k);
        return it != m.end() ? it->second : def;
    }

    static std::string canonicalize(const std::string& raw) {
        std::string s = raw;
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return std::tolower(c); });

        static const std::map<std::string, std::string> aliases = {
            // Coding & file editing
            {"read_file", "read_workspace_file"},
            {"read_workspace_file", "read_workspace_file"},
            {"write_file", "write_workspace_file"},
            {"write_workspace_file", "write_workspace_file"},
            {"create_file", "write_workspace_file"},
            {"patch_file", "patch_workspace_file"},
            {"patch_workspace_file", "patch_workspace_file"},
            {"edit_lines", "edit_workspace_lines"},
            {"edit_workspace_lines", "edit_workspace_lines"},
            {"list_files", "list_workspace_files"},
            {"list_workspace_files", "list_workspace_files"},

            // Code search
            {"grep", "grep_workspace"},
            {"ggrep", "grep_workspace"},
            {"grep_workspace", "grep_workspace"},
            {"search_code", "grep_workspace"},
            {"find_files", "find_workspace_files"},
            {"find_workspace_files", "find_workspace_files"},
            {"outline", "outline_python_file"},
            {"outline_file", "outline_python_file"},
            {"outline_python_file", "outline_python_file"},
            {"find_symbol", "find_symbol_definitions"},
            {"find_symbol_definitions", "find_symbol_definitions"},

            // Syntax checking & debugging
            {"check_syntax", "check_python_file"},
            {"check_python", "check_python_file"},
            {"check_python_file", "check_python_file"},
            {"diagnose_error", "diagnose_tool_error"},
            {"diagnose_tool_error", "diagnose_tool_error"},
            {"check_project", "run_project_checks"},
            {"run_project_checks", "run_project_checks"},

            // Terminal & Git
            {"terminal", "execute_terminal"},
            {"bash", "execute_terminal"},
            {"sh", "execute_terminal"},
            {"execute_terminal", "execute_terminal"},
            {"git", "git_status"},
            {"git_status", "git_status"},
            {"git_diff", "git_diff"},

            // Diagnostics
            {"check_port", "check_port_open"},
            {"check_port_open", "check_port_open"},
            {"permissions", "inspect_workspace_permissions"},
            {"inspect_workspace_permissions", "inspect_workspace_permissions"},
            {"telemetry", "get_system_telemetry"},
            {"get_system_telemetry", "get_system_telemetry"}
        };
        auto it = aliases.find(s);
        return it != aliases.end() ? it->second : s;
    }
};

} // namespace moeboost
