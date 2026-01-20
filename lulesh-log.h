#ifndef LULESH_LOG_H
#define LULESH_LOG_H

#include <cerrno>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>

#include <sys/stat.h>
#include <sys/types.h>

namespace lulesh_log {

inline std::string JoinPath(const std::string& left, const std::string& right)
{
  if (left.empty()) {
    return right;
  }
  if (right.empty()) {
    return left;
  }
  if (left.back() == '/') {
    return left + right;
  }
  return left + "/" + right;
}

inline bool MakeDir(const std::string& path)
{
  if (path.empty()) {
    return false;
  }
  if (::mkdir(path.c_str(), 0755) == 0) {
    return true;
  }
  return errno == EEXIST;
}

inline bool EnsureDir(const std::string& path)
{
  if (path.empty()) {
    return false;
  }

  std::string current;
  std::size_t start = 0;
  if (path[0] == '/') {
    current = "/";
    start = 1;
  }

  while (start <= path.size()) {
    std::size_t slash = path.find('/', start);
    std::size_t len = (slash == std::string::npos) ? path.size() - start : slash - start;
    std::string part = path.substr(start, len);
    if (!part.empty()) {
      if (current.size() > 1 && current.back() != '/') {
        current += "/";
      }
      current += part;
      if (!MakeDir(current)) {
        return false;
      }
    }
    if (slash == std::string::npos) {
      break;
    }
    start = slash + 1;
  }

  return true;
}

inline std::string DefaultLogRoot()
{
  return "benchmarks/logs";
}

inline std::string MakeStepDir(const std::string& base_dir,
                               const std::string& step_name,
                               int rank)
{
  std::ostringstream name;
  name << step_name << "_rank" << rank;
  std::string dir = JoinPath(base_dir, name.str());
  EnsureDir(dir);
  return dir;
}

inline std::string MakeMatrixDir(const std::string& step_dir)
{
  std::string dir = JoinPath(step_dir, "matrix");
  EnsureDir(dir);
  return dir;
}

inline std::string MakeInfoDir(const std::string& step_dir)
{
  std::string dir = JoinPath(step_dir, "info");
  EnsureDir(dir);
  return dir;
}

template <typename T>
inline void SetStreamPrecision(std::ofstream& out)
{
  if (std::is_floating_point<T>::value) {
    out.setf(std::ios::scientific);
    out << std::setprecision(std::numeric_limits<T>::max_digits10);
  }
}

template <typename T>
inline bool WriteCsvScalar(const std::string& path, const T& value)
{
  std::ofstream out(path.c_str());
  if (!out) {
    return false;
  }
  SetStreamPrecision<T>(out);
  out << value << "\n";
  return true;
}

template <typename T>
inline bool WriteCsvArray(const std::string& path,
                          const T* data,
                          std::size_t count,
                          std::size_t stride = 1)
{
  if (data == NULL || count == 0 || stride == 0) {
    return false;
  }
  std::ofstream out(path.c_str());
  if (!out) {
    return false;
  }
  SetStreamPrecision<T>(out);
  for (std::size_t i = 0; i < count; i += stride) {
    out << data[i];
    if (i + stride < count) {
      out << "\n";
    }
  }
  return true;
}

template <typename T>
inline bool WriteCsvArray3(const std::string& path,
                           const T* a,
                           const T* b,
                           const T* c,
                           std::size_t count)
{
  if (a == NULL || b == NULL || c == NULL || count == 0) {
    return false;
  }
  std::ofstream out(path.c_str());
  if (!out) {
    return false;
  }
  SetStreamPrecision<T>(out);
  for (std::size_t i = 0; i < count; ++i) {
    out << a[i] << "," << b[i] << "," << c[i];
    if (i + 1 < count) {
      out << "\n";
    }
  }
  return true;
}

} // namespace lulesh_log

#endif // LULESH_LOG_H
