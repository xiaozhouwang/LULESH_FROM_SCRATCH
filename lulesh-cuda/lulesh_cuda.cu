#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>

#include "../lulesh-log.h"
#include "../lulesh.h"

// Forward declarations from existing CPU utilities.
void ParseCommandLineOptions(int argc, char* argv[], int myRank, cmdLineOpts* opts);
void InitMeshDecomp(Int_t numRanks, Int_t myRank,
                    Int_t* col, Int_t* row, Int_t* plane, Int_t* side);
void DumpToVisit(Domain& domain, Int_t numFiles, Int_t myRank, Int_t numRanks);
void VerifyAndWriteFinalOutput(Real_t elapsed_time, Domain& locDom,
                               Int_t nx, Int_t numRanks);

static int g_myRank = 0;
static int g_numRanks = 1;
static cmdLineOpts g_cmdline_opts;
static bool g_cmdline_opts_set = false;

struct LogConfig {
  bool enabled;
  bool log_pre;
  bool log_substeps;
  Index_t max_cycle;
  Index_t cycle_stride;
  Index_t stride;
  std::string root;
  std::set<std::string> fields;
  std::set<Index_t> cycle_list;

  LogConfig()
      : enabled(false),
        log_pre(false),
        log_substeps(false),
        max_cycle(1),
        cycle_stride(1),
        stride(1),
        root(lulesh_log::DefaultLogRoot())
  {
  }
};

static bool EnvEnabled(const char* name)
{
  const char* value = getenv(name);
  if (value == NULL || value[0] == '\0') {
    return false;
  }
  return strcmp(value, "0") != 0;
}

static Index_t ParseStride(const char* name)
{
  const char* value = getenv(name);
  if (value == NULL || value[0] == '\0') {
    return 1;
  }
  char* end = NULL;
  long parsed = strtol(value, &end, 10);
  if (end == value || parsed <= 0) {
    return 1;
  }
  if (parsed > std::numeric_limits<Index_t>::max()) {
    return 1;
  }
  return static_cast<Index_t>(parsed);
}

static Index_t ParseMaxCycle(const char* name)
{
  const char* value = getenv(name);
  if (value == NULL || value[0] == '\0') {
    return 1;
  }
  char* end = NULL;
  long parsed = strtol(value, &end, 10);
  if (end == value || parsed <= 0) {
    return 1;
  }
  if (parsed > std::numeric_limits<Index_t>::max()) {
    return 1;
  }
  return static_cast<Index_t>(parsed);
}

static std::string TrimCopy(const std::string& input)
{
  std::string result = input;
  std::string::size_type start = result.find_first_not_of(" \t\r\n");
  if (start == std::string::npos) {
    return "";
  }
  std::string::size_type end = result.find_last_not_of(" \t\r\n");
  result = result.substr(start, end - start + 1);
  return result;
}

static std::set<Index_t> ParseCycleList(const char* name)
{
  std::set<Index_t> cycles;
  const char* value = getenv(name);
  if (value == NULL || value[0] == '\0') {
    return cycles;
  }
  std::string raw = TrimCopy(value);
  if (raw.empty()) {
    return cycles;
  }
  std::string::size_type start = 0;
  while (start < raw.size()) {
    std::string::size_type comma = raw.find(',', start);
    std::string token = raw.substr(start,
                                   (comma == std::string::npos) ? std::string::npos
                                                                : comma - start);
    token = TrimCopy(token);
    if (!token.empty()) {
      char* end = NULL;
      long parsed = strtol(token.c_str(), &end, 10);
      if (end != token.c_str() && parsed > 0 &&
          parsed <= std::numeric_limits<Index_t>::max()) {
        cycles.insert(static_cast<Index_t>(parsed));
      }
    }
    if (comma == std::string::npos) {
      break;
    }
    start = comma + 1;
  }
  return cycles;
}

static std::set<std::string> ParseFields()
{
  std::set<std::string> fields;
  const char* value = getenv("LULESH_LOG_FIELDS");
  if (value == NULL || value[0] == '\0') {
    return fields;
  }
  std::string raw = TrimCopy(value);
  if (raw.empty()) {
    return fields;
  }
  std::string::size_type start = 0;
  while (start < raw.size()) {
    std::string::size_type comma = raw.find(',', start);
    std::string token = raw.substr(start,
                                   (comma == std::string::npos) ? std::string::npos
                                                                : comma - start);
    token = TrimCopy(token);
    if (!token.empty()) {
      fields.insert(token);
    }
    if (comma == std::string::npos) {
      break;
    }
    start = comma + 1;
  }
  return fields;
}

static const LogConfig& GetLogConfig()
{
  static LogConfig cfg;
  static bool initialized = false;
  if (!initialized) {
    cfg.enabled = EnvEnabled("LULESH_LOG_ENABLE");
    cfg.log_pre = EnvEnabled("LULESH_LOG_PRE");
    cfg.log_substeps = EnvEnabled("LULESH_LOG_SUBSTEPS");
    cfg.max_cycle = ParseMaxCycle("LULESH_LOG_CYCLES");
    cfg.cycle_stride = ParseStride("LULESH_LOG_CYCLE_STRIDE");
    cfg.stride = ParseStride("LULESH_LOG_STRIDE");
    cfg.cycle_list = ParseCycleList("LULESH_LOG_CYCLE_LIST");
    cfg.fields = ParseFields();
    const char* root = getenv("LULESH_LOG_ROOT");
    if (root != NULL && root[0] != '\0') {
      cfg.root = root;
    }
    initialized = true;
  }
  return cfg;
}

static bool ShouldLogField(const LogConfig& cfg, const std::string& name)
{
  if (cfg.fields.empty()) {
    return true;
  }
  return cfg.fields.find(name) != cfg.fields.end();
}

static bool ShouldLogStep(const LogConfig& cfg, Domain& domain)
{
  if (!cfg.enabled) {
    return false;
  }
  Index_t cycle = domain.cycle();
  if (cycle < 1) {
    return false;
  }
  if (!cfg.cycle_list.empty()) {
    return cfg.cycle_list.find(cycle) != cfg.cycle_list.end();
  }
  if (cycle > cfg.max_cycle) {
    return false;
  }
  if (cfg.cycle_stride <= 1) {
    return true;
  }
  return ((cycle - 1) % cfg.cycle_stride) == 0;
}

static std::string StepNameWithCycle(const std::string& base, Index_t cycle)
{
  std::ostringstream name;
  name << base << "_cycle" << cycle;
  return name.str();
}

static std::string JoinCsvPath(const std::string& dir, const std::string& name)
{
  return lulesh_log::JoinPath(dir, name + ".csv");
}

static std::string GetGitHash()
{
#ifdef LULESH_GIT_HASH
  return LULESH_GIT_HASH;
#else
  const char* value = getenv("LULESH_GIT_HASH");
  if (value != NULL && value[0] != '\0') {
    return value;
  }
  return "unknown";
#endif
}

static std::string GetEnvString(const char* name)
{
  const char* value = getenv(name);
  if (value != NULL) {
    return value;
  }
  return "";
}

static void WriteInfoFile(const LogConfig& cfg,
                          Domain& domain,
                          const std::string& step_name,
                          const std::string& info_dir)
{
  std::string info_path = lulesh_log::JoinPath(info_dir, "info.md");
  std::ofstream out(info_path.c_str());
  if (!out) {
    return;
  }

  time_t now = time(NULL);
  char time_buf[64];
  tm* utc = gmtime(&now);
  if (utc != NULL) {
    strftime(time_buf, sizeof(time_buf), "%Y-%m-%dT%H:%M:%SZ", utc);
  } else {
    std::strcpy(time_buf, "unknown");
  }

  out << "step: " << step_name << "\n";
  out << "timestamp_utc: " << time_buf << "\n";
  out << "rank: " << g_myRank << "\n";
  out << "numRanks: " << g_numRanks << "\n";
  out << "omp_max_threads: 1\n";
  out << "omp_env_threads: " << GetEnvString("OMP_NUM_THREADS") << "\n";

  if (g_cmdline_opts_set) {
    out << "nx: " << g_cmdline_opts.nx << "\n";
    out << "its: " << g_cmdline_opts.its << "\n";
    out << "numReg: " << g_cmdline_opts.numReg << "\n";
    out << "numFiles: " << g_cmdline_opts.numFiles << "\n";
    out << "showProg: " << g_cmdline_opts.showProg << "\n";
    out << "quiet: " << g_cmdline_opts.quiet << "\n";
    out << "viz: " << g_cmdline_opts.viz << "\n";
    out << "cost: " << g_cmdline_opts.cost << "\n";
    out << "balance: " << g_cmdline_opts.balance << "\n";
  }

  out << "sizeX: " << domain.sizeX() << "\n";
  out << "sizeY: " << domain.sizeY() << "\n";
  out << "sizeZ: " << domain.sizeZ() << "\n";
  out << "colLoc: " << domain.colLoc() << "\n";
  out << "rowLoc: " << domain.rowLoc() << "\n";
  out << "planeLoc: " << domain.planeLoc() << "\n";
  out << "tp: " << domain.tp() << "\n";
  out << "numElem: " << domain.numElem() << "\n";
  out << "numNode: " << domain.numNode() << "\n";
  out << "Real_t_bytes: " << sizeof(Real_t) << "\n";
  out << "Real_t_max_digits10: " << std::numeric_limits<Real_t>::max_digits10
      << "\n";
  out << "compiler: " << __VERSION__ << "\n";
#ifdef __OPTIMIZE__
  out << "optimize: true\n";
#else
  out << "optimize: false\n";
#endif
  out << "openmp: 0\n";
  out << "use_mpi: " << USE_MPI << "\n";
  out << "git_hash: " << GetGitHash() << "\n";
  out << "build_flags_env: " << GetEnvString("LULESH_BUILD_FLAGS") << "\n";
  out << "log_root: " << cfg.root << "\n";
  out << "log_stride: " << cfg.stride << "\n";
  out << "log_cycles: " << cfg.max_cycle << "\n";
  out << "log_cycle_stride: " << cfg.cycle_stride << "\n";
  out << "log_enabled: " << (cfg.enabled ? "true" : "false") << "\n";
  out << "log_pre: " << (cfg.log_pre ? "true" : "false") << "\n";
  out << "log_substeps: " << (cfg.log_substeps ? "true" : "false") << "\n";
  out << "cycle: " << domain.cycle() << "\n";

  if (!cfg.cycle_list.empty()) {
    std::ostringstream joined;
    for (std::set<Index_t>::const_iterator it = cfg.cycle_list.begin();
         it != cfg.cycle_list.end(); ++it) {
      if (it != cfg.cycle_list.begin()) {
        joined << ",";
      }
      joined << *it;
    }
    out << "log_cycle_list: " << joined.str() << "\n";
  } else {
    out << "log_cycle_list: none\n";
  }

  if (!cfg.fields.empty()) {
    std::ostringstream joined;
    for (std::set<std::string>::const_iterator it = cfg.fields.begin();
         it != cfg.fields.end(); ++it) {
      if (it != cfg.fields.begin()) {
        joined << ",";
      }
      joined << *it;
    }
    out << "log_fields: " << joined.str() << "\n";
  } else {
    out << "log_fields: all\n";
  }
}

static bool WriteCsvFromAccessor(const std::string& path,
                                 Domain& domain,
                                 Index_t count,
                                 Index_t stride,
                                 Real_t& (Domain::*accessor)(Index_t))
{
  if (count <= 0 || stride <= 0) {
    return false;
  }
  std::ofstream out(path.c_str());
  if (!out) {
    return false;
  }
  lulesh_log::SetStreamPrecision<Real_t>(out);
  for (Index_t i = 0; i < count; i += stride) {
    out << (domain.*accessor)(i);
    if (i + stride < count) {
      out << "\n";
    }
  }
  return true;
}

static void LogNodalFields(const LogConfig& cfg,
                           Domain& domain,
                           const std::string& step_name)
{
  std::string step_dir = lulesh_log::MakeStepDir(cfg.root, step_name, g_myRank);
  std::string matrix_dir = lulesh_log::MakeMatrixDir(step_dir);
  std::string info_dir = lulesh_log::MakeInfoDir(step_dir);
  WriteInfoFile(cfg, domain, step_name, info_dir);

  Index_t count = domain.numNode();
  Index_t stride = cfg.stride;

  if (ShouldLogField(cfg, "x")) {
    WriteCsvFromAccessor(JoinCsvPath(matrix_dir, "x"),
                         domain, count, stride, &Domain::x);
  }
  if (ShouldLogField(cfg, "y")) {
    WriteCsvFromAccessor(JoinCsvPath(matrix_dir, "y"),
                         domain, count, stride, &Domain::y);
  }
  if (ShouldLogField(cfg, "z")) {
    WriteCsvFromAccessor(JoinCsvPath(matrix_dir, "z"),
                         domain, count, stride, &Domain::z);
  }
  if (ShouldLogField(cfg, "xd")) {
    WriteCsvFromAccessor(JoinCsvPath(matrix_dir, "xd"),
                         domain, count, stride, &Domain::xd);
  }
  if (ShouldLogField(cfg, "yd")) {
    WriteCsvFromAccessor(JoinCsvPath(matrix_dir, "yd"),
                         domain, count, stride, &Domain::yd);
  }
  if (ShouldLogField(cfg, "zd")) {
    WriteCsvFromAccessor(JoinCsvPath(matrix_dir, "zd"),
                         domain, count, stride, &Domain::zd);
  }
  if (ShouldLogField(cfg, "xdd")) {
    WriteCsvFromAccessor(JoinCsvPath(matrix_dir, "xdd"),
                         domain, count, stride, &Domain::xdd);
  }
  if (ShouldLogField(cfg, "ydd")) {
    WriteCsvFromAccessor(JoinCsvPath(matrix_dir, "ydd"),
                         domain, count, stride, &Domain::ydd);
  }
  if (ShouldLogField(cfg, "zdd")) {
    WriteCsvFromAccessor(JoinCsvPath(matrix_dir, "zdd"),
                         domain, count, stride, &Domain::zdd);
  }
  if (ShouldLogField(cfg, "fx")) {
    WriteCsvFromAccessor(JoinCsvPath(matrix_dir, "fx"),
                         domain, count, stride, &Domain::fx);
  }
  if (ShouldLogField(cfg, "fy")) {
    WriteCsvFromAccessor(JoinCsvPath(matrix_dir, "fy"),
                         domain, count, stride, &Domain::fy);
  }
  if (ShouldLogField(cfg, "fz")) {
    WriteCsvFromAccessor(JoinCsvPath(matrix_dir, "fz"),
                         domain, count, stride, &Domain::fz);
  }
  if (ShouldLogField(cfg, "nodalMass")) {
    WriteCsvFromAccessor(JoinCsvPath(matrix_dir, "nodalMass"),
                         domain, count, stride, &Domain::nodalMass);
  }
}

static void LogElementFields(const LogConfig& cfg,
                             Domain& domain,
                             const std::string& step_name)
{
  std::string step_dir = lulesh_log::MakeStepDir(cfg.root, step_name, g_myRank);
  std::string matrix_dir = lulesh_log::MakeMatrixDir(step_dir);
  std::string info_dir = lulesh_log::MakeInfoDir(step_dir);
  WriteInfoFile(cfg, domain, step_name, info_dir);

  Index_t count = domain.numElem();
  Index_t stride = cfg.stride;

  if (ShouldLogField(cfg, "e")) {
    WriteCsvFromAccessor(JoinCsvPath(matrix_dir, "e"),
                         domain, count, stride, &Domain::e);
  }
  if (ShouldLogField(cfg, "p")) {
    WriteCsvFromAccessor(JoinCsvPath(matrix_dir, "p"),
                         domain, count, stride, &Domain::p);
  }
  if (ShouldLogField(cfg, "q")) {
    WriteCsvFromAccessor(JoinCsvPath(matrix_dir, "q"),
                         domain, count, stride, &Domain::q);
  }
  if (ShouldLogField(cfg, "ql")) {
    WriteCsvFromAccessor(JoinCsvPath(matrix_dir, "ql"),
                         domain, count, stride, &Domain::ql);
  }
  if (ShouldLogField(cfg, "qq")) {
    WriteCsvFromAccessor(JoinCsvPath(matrix_dir, "qq"),
                         domain, count, stride, &Domain::qq);
  }
  if (ShouldLogField(cfg, "v")) {
    WriteCsvFromAccessor(JoinCsvPath(matrix_dir, "v"),
                         domain, count, stride, &Domain::v);
  }
  if (ShouldLogField(cfg, "vnew")) {
    WriteCsvFromAccessor(JoinCsvPath(matrix_dir, "vnew"),
                         domain, count, stride, &Domain::vnew);
  }
  if (ShouldLogField(cfg, "volo")) {
    WriteCsvFromAccessor(JoinCsvPath(matrix_dir, "volo"),
                         domain, count, stride, &Domain::volo);
  }
  if (ShouldLogField(cfg, "delv")) {
    WriteCsvFromAccessor(JoinCsvPath(matrix_dir, "delv"),
                         domain, count, stride, &Domain::delv);
  }
  if (ShouldLogField(cfg, "vdov")) {
    WriteCsvFromAccessor(JoinCsvPath(matrix_dir, "vdov"),
                         domain, count, stride, &Domain::vdov);
  }
  if (ShouldLogField(cfg, "arealg")) {
    WriteCsvFromAccessor(JoinCsvPath(matrix_dir, "arealg"),
                         domain, count, stride, &Domain::arealg);
  }
  if (ShouldLogField(cfg, "ss")) {
    WriteCsvFromAccessor(JoinCsvPath(matrix_dir, "ss"),
                         domain, count, stride, &Domain::ss);
  }
  if (ShouldLogField(cfg, "elemMass")) {
    WriteCsvFromAccessor(JoinCsvPath(matrix_dir, "elemMass"),
                         domain, count, stride, &Domain::elemMass);
  }
}

static void LogHourglassArrays(const LogConfig& cfg,
                               Domain& domain,
                               const std::string& step_name,
                               const Real_t* dvdx,
                               const Real_t* dvdy,
                               const Real_t* dvdz,
                               const Real_t* x8n,
                               const Real_t* y8n,
                               const Real_t* z8n,
                               Index_t count)
{
  bool any = ShouldLogField(cfg, "dvdx") || ShouldLogField(cfg, "dvdy") ||
             ShouldLogField(cfg, "dvdz") || ShouldLogField(cfg, "x8n") ||
             ShouldLogField(cfg, "y8n") || ShouldLogField(cfg, "z8n");
  if (!any) {
    return;
  }

  std::string step_dir = lulesh_log::MakeStepDir(cfg.root, step_name, g_myRank);
  std::string matrix_dir = lulesh_log::MakeMatrixDir(step_dir);
  std::string info_dir = lulesh_log::MakeInfoDir(step_dir);
  WriteInfoFile(cfg, domain, step_name, info_dir);

  std::size_t size = static_cast<std::size_t>(count);
  std::size_t stride = static_cast<std::size_t>(cfg.stride);

  if (ShouldLogField(cfg, "dvdx")) {
    lulesh_log::WriteCsvArray(JoinCsvPath(matrix_dir, "dvdx"),
                              dvdx, size, stride);
  }
  if (ShouldLogField(cfg, "dvdy")) {
    lulesh_log::WriteCsvArray(JoinCsvPath(matrix_dir, "dvdy"),
                              dvdy, size, stride);
  }
  if (ShouldLogField(cfg, "dvdz")) {
    lulesh_log::WriteCsvArray(JoinCsvPath(matrix_dir, "dvdz"),
                              dvdz, size, stride);
  }
  if (ShouldLogField(cfg, "x8n")) {
    lulesh_log::WriteCsvArray(JoinCsvPath(matrix_dir, "x8n"),
                              x8n, size, stride);
  }
  if (ShouldLogField(cfg, "y8n")) {
    lulesh_log::WriteCsvArray(JoinCsvPath(matrix_dir, "y8n"),
                              y8n, size, stride);
  }
  if (ShouldLogField(cfg, "z8n")) {
    lulesh_log::WriteCsvArray(JoinCsvPath(matrix_dir, "z8n"),
                              z8n, size, stride);
  }
}

static void LogHourglassForces(const LogConfig& cfg,
                               Domain& domain,
                               const std::string& step_name,
                               const Real_t* hgfx,
                               const Real_t* hgfy,
                               const Real_t* hgfz,
                               Index_t count)
{
  bool any = ShouldLogField(cfg, "hgfx") ||
             ShouldLogField(cfg, "hgfy") ||
             ShouldLogField(cfg, "hgfz");
  if (!any) {
    return;
  }

  std::string step_dir = lulesh_log::MakeStepDir(cfg.root, step_name, g_myRank);
  std::string matrix_dir = lulesh_log::MakeMatrixDir(step_dir);
  std::string info_dir = lulesh_log::MakeInfoDir(step_dir);
  WriteInfoFile(cfg, domain, step_name, info_dir);

  std::size_t size = static_cast<std::size_t>(count);
  std::size_t stride = static_cast<std::size_t>(cfg.stride);

  if (ShouldLogField(cfg, "hgfx")) {
    lulesh_log::WriteCsvArray(JoinCsvPath(matrix_dir, "hgfx"),
                              hgfx, size, stride);
  }
  if (ShouldLogField(cfg, "hgfy")) {
    lulesh_log::WriteCsvArray(JoinCsvPath(matrix_dir, "hgfy"),
                              hgfy, size, stride);
  }
  if (ShouldLogField(cfg, "hgfz")) {
    lulesh_log::WriteCsvArray(JoinCsvPath(matrix_dir, "hgfz"),
                              hgfz, size, stride);
  }
}

static void LogTimeConstraintFields(const LogConfig& cfg,
                                    Domain& domain,
                                    const std::string& step_name)
{
  std::string step_dir = lulesh_log::MakeStepDir(cfg.root, step_name, g_myRank);
  std::string matrix_dir = lulesh_log::MakeMatrixDir(step_dir);
  std::string info_dir = lulesh_log::MakeInfoDir(step_dir);
  WriteInfoFile(cfg, domain, step_name, info_dir);

  if (ShouldLogField(cfg, "deltatime")) {
    lulesh_log::WriteCsvScalar(JoinCsvPath(matrix_dir, "deltatime"),
                               domain.deltatime());
  }
  if (ShouldLogField(cfg, "dtcourant")) {
    lulesh_log::WriteCsvScalar(JoinCsvPath(matrix_dir, "dtcourant"),
                               domain.dtcourant());
  }
  if (ShouldLogField(cfg, "dthydro")) {
    lulesh_log::WriteCsvScalar(JoinCsvPath(matrix_dir, "dthydro"),
                               domain.dthydro());
  }
}

static inline void TimeIncrement(Domain& domain)
{
  Real_t targetdt = domain.stoptime() - domain.time();

  if ((domain.dtfixed() <= Real_t(0.0)) && (domain.cycle() != Int_t(0))) {
    Real_t ratio;
    Real_t olddt = domain.deltatime();

    Real_t gnewdt = Real_t(1.0e+20);
    Real_t newdt;
    if (domain.dtcourant() < gnewdt) {
      gnewdt = domain.dtcourant() / Real_t(2.0);
    }
    if (domain.dthydro() < gnewdt) {
      gnewdt = domain.dthydro() * Real_t(2.0) / Real_t(3.0);
    }

    newdt = gnewdt;

    ratio = newdt / olddt;
    if (ratio >= Real_t(1.0)) {
      if (ratio < domain.deltatimemultlb()) {
        newdt = olddt;
      } else if (ratio > domain.deltatimemultub()) {
        newdt = olddt * domain.deltatimemultub();
      }
    }

    if (newdt > domain.dtmax()) {
      newdt = domain.dtmax();
    }
    domain.deltatime() = newdt;
  }

  if ((targetdt > domain.deltatime()) &&
      (targetdt < (Real_t(4.0) * domain.deltatime() / Real_t(3.0)))) {
    targetdt = Real_t(2.0) * domain.deltatime() / Real_t(3.0);
  }

  if (targetdt < domain.deltatime()) {
    domain.deltatime() = targetdt;
  }

  domain.time() += domain.deltatime();

  ++domain.cycle();
}

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t err = (call);                                                \
    if (err != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error %s at %s:%d\n",                   \
                   cudaGetErrorString(err), __FILE__, __LINE__);            \
      std::exit(1);                                                         \
    }                                                                        \
  } while (0)

__host__ __device__ inline Real_t CalcElemVolume24(
    const Real_t x0, const Real_t x1, const Real_t x2, const Real_t x3,
    const Real_t x4, const Real_t x5, const Real_t x6, const Real_t x7,
    const Real_t y0, const Real_t y1, const Real_t y2, const Real_t y3,
    const Real_t y4, const Real_t y5, const Real_t y6, const Real_t y7,
    const Real_t z0, const Real_t z1, const Real_t z2, const Real_t z3,
    const Real_t z4, const Real_t z5, const Real_t z6, const Real_t z7)
{
  Real_t twelveth = Real_t(1.0) / Real_t(12.0);

  Real_t dx61 = x6 - x1;
  Real_t dy61 = y6 - y1;
  Real_t dz61 = z6 - z1;

  Real_t dx70 = x7 - x0;
  Real_t dy70 = y7 - y0;
  Real_t dz70 = z7 - z0;

  Real_t dx63 = x6 - x3;
  Real_t dy63 = y6 - y3;
  Real_t dz63 = z6 - z3;

  Real_t dx20 = x2 - x0;
  Real_t dy20 = y2 - y0;
  Real_t dz20 = z2 - z0;

  Real_t dx50 = x5 - x0;
  Real_t dy50 = y5 - y0;
  Real_t dz50 = z5 - z0;

  Real_t dx64 = x6 - x4;
  Real_t dy64 = y6 - y4;
  Real_t dz64 = z6 - z4;

  Real_t dx31 = x3 - x1;
  Real_t dy31 = y3 - y1;
  Real_t dz31 = z3 - z1;

  Real_t dx72 = x7 - x2;
  Real_t dy72 = y7 - y2;
  Real_t dz72 = z7 - z2;

  Real_t dx43 = x4 - x3;
  Real_t dy43 = y4 - y3;
  Real_t dz43 = z4 - z3;

  Real_t dx57 = x5 - x7;
  Real_t dy57 = y5 - y7;
  Real_t dz57 = z5 - z7;

  Real_t dx14 = x1 - x4;
  Real_t dy14 = y1 - y4;
  Real_t dz14 = z1 - z4;

  Real_t dx25 = x2 - x5;
  Real_t dy25 = y2 - y5;
  Real_t dz25 = z2 - z5;

#define TRIPLE_PRODUCT(x1_, y1_, z1_, x2_, y2_, z2_, x3_, y3_, z3_) \
  ((x1_) * ((y2_) * (z3_) - (z2_) * (y3_)) +                         \
   (x2_) * ((z1_) * (y3_) - (y1_) * (z3_)) +                         \
   (x3_) * ((y1_) * (z2_) - (z1_) * (y2_)))

  Real_t volume =
      TRIPLE_PRODUCT(dx31 + dx72, dx63, dx20, dy31 + dy72, dy63, dy20,
                     dz31 + dz72, dz63, dz20) +
      TRIPLE_PRODUCT(dx43 + dx57, dx64, dx70, dy43 + dy57, dy64, dy70,
                     dz43 + dz57, dz64, dz70) +
      TRIPLE_PRODUCT(dx14 + dx25, dx61, dx50, dy14 + dy25, dy61, dy50,
                     dz14 + dz25, dz61, dz50);

#undef TRIPLE_PRODUCT

  volume *= twelveth;

  return volume;
}

__host__ __device__ inline Real_t CalcElemVolumeDevice(const Real_t x[8],
                                                       const Real_t y[8],
                                                       const Real_t z[8])
{
  return CalcElemVolume24(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7],
                          y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7],
                          z[0], z[1], z[2], z[3], z[4], z[5], z[6], z[7]);
}

// Host-visible definition for lulesh-init.cc
Real_t CalcElemVolume(const Real_t x[8], const Real_t y[8], const Real_t z[8])
{
  return CalcElemVolume24(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7],
                          y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7],
                          z[0], z[1], z[2], z[3], z[4], z[5], z[6], z[7]);
}

__device__ inline Real_t DeviceMax(Real_t a, Real_t b)
{
  return a > b ? a : b;
}

__device__ inline void CalcElemShapeFunctionDerivatives(const Real_t x[8],
                                                        const Real_t y[8],
                                                        const Real_t z[8],
                                                        Real_t b[3][8],
                                                        Real_t* volume)
{
  const Real_t x0 = x[0];
  const Real_t x1 = x[1];
  const Real_t x2 = x[2];
  const Real_t x3 = x[3];
  const Real_t x4 = x[4];
  const Real_t x5 = x[5];
  const Real_t x6 = x[6];
  const Real_t x7 = x[7];

  const Real_t y0 = y[0];
  const Real_t y1 = y[1];
  const Real_t y2 = y[2];
  const Real_t y3 = y[3];
  const Real_t y4 = y[4];
  const Real_t y5 = y[5];
  const Real_t y6 = y[6];
  const Real_t y7 = y[7];

  const Real_t z0 = z[0];
  const Real_t z1 = z[1];
  const Real_t z2 = z[2];
  const Real_t z3 = z[3];
  const Real_t z4 = z[4];
  const Real_t z5 = z[5];
  const Real_t z6 = z[6];
  const Real_t z7 = z[7];

  Real_t fjxxi = Real_t(.125) * ((x6 - x0) + (x5 - x3) - (x7 - x1) - (x4 - x2));
  Real_t fjxet = Real_t(.125) * ((x6 - x0) - (x5 - x3) + (x7 - x1) - (x4 - x2));
  Real_t fjxze = Real_t(.125) * ((x6 - x0) + (x5 - x3) + (x7 - x1) + (x4 - x2));

  Real_t fjyxi = Real_t(.125) * ((y6 - y0) + (y5 - y3) - (y7 - y1) - (y4 - y2));
  Real_t fjyet = Real_t(.125) * ((y6 - y0) - (y5 - y3) + (y7 - y1) - (y4 - y2));
  Real_t fjyze = Real_t(.125) * ((y6 - y0) + (y5 - y3) + (y7 - y1) + (y4 - y2));

  Real_t fjzxi = Real_t(.125) * ((z6 - z0) + (z5 - z3) - (z7 - z1) - (z4 - z2));
  Real_t fjzet = Real_t(.125) * ((z6 - z0) - (z5 - z3) + (z7 - z1) - (z4 - z2));
  Real_t fjzze = Real_t(.125) * ((z6 - z0) + (z5 - z3) + (z7 - z1) + (z4 - z2));

  Real_t cjxxi = (fjyet * fjzze) - (fjzet * fjyze);
  Real_t cjxet = -(fjyxi * fjzze) + (fjzxi * fjyze);
  Real_t cjxze = (fjyxi * fjzet) - (fjzxi * fjyet);

  Real_t cjyxi = -(fjxet * fjzze) + (fjzet * fjxze);
  Real_t cjyet = (fjxxi * fjzze) - (fjzxi * fjxze);
  Real_t cjyze = -(fjxxi * fjzet) + (fjzxi * fjxet);

  Real_t cjzxi = (fjxet * fjyze) - (fjyet * fjxze);
  Real_t cjzet = -(fjxxi * fjyze) + (fjyxi * fjxze);
  Real_t cjzze = (fjxxi * fjyet) - (fjyxi * fjxet);

  b[0][0] = -cjxxi - cjxet - cjxze;
  b[0][1] = cjxxi - cjxet - cjxze;
  b[0][2] = cjxxi + cjxet - cjxze;
  b[0][3] = -cjxxi + cjxet - cjxze;
  b[0][4] = -b[0][2];
  b[0][5] = -b[0][3];
  b[0][6] = -b[0][0];
  b[0][7] = -b[0][1];

  b[1][0] = -cjyxi - cjyet - cjyze;
  b[1][1] = cjyxi - cjyet - cjyze;
  b[1][2] = cjyxi + cjyet - cjyze;
  b[1][3] = -cjyxi + cjyet - cjyze;
  b[1][4] = -b[1][2];
  b[1][5] = -b[1][3];
  b[1][6] = -b[1][0];
  b[1][7] = -b[1][1];

  b[2][0] = -cjzxi - cjzet - cjzze;
  b[2][1] = cjzxi - cjzet - cjzze;
  b[2][2] = cjzxi + cjzet - cjzze;
  b[2][3] = -cjzxi + cjzet - cjzze;
  b[2][4] = -b[2][2];
  b[2][5] = -b[2][3];
  b[2][6] = -b[2][0];
  b[2][7] = -b[2][1];

  *volume = Real_t(8.0) * (fjxet * cjxet + fjyet * cjyet + fjzet * cjzet);
}

__device__ inline void SumElemFaceNormal(Real_t* normalX0, Real_t* normalY0,
                                         Real_t* normalZ0, Real_t* normalX1,
                                         Real_t* normalY1, Real_t* normalZ1,
                                         Real_t* normalX2, Real_t* normalY2,
                                         Real_t* normalZ2, Real_t* normalX3,
                                         Real_t* normalY3, Real_t* normalZ3,
                                         const Real_t x0, const Real_t y0,
                                         const Real_t z0, const Real_t x1,
                                         const Real_t y1, const Real_t z1,
                                         const Real_t x2, const Real_t y2,
                                         const Real_t z2, const Real_t x3,
                                         const Real_t y3, const Real_t z3)
{
  Real_t bisectX0 = Real_t(0.5) * (x3 + x2 - x1 - x0);
  Real_t bisectY0 = Real_t(0.5) * (y3 + y2 - y1 - y0);
  Real_t bisectZ0 = Real_t(0.5) * (z3 + z2 - z1 - z0);
  Real_t bisectX1 = Real_t(0.5) * (x2 + x1 - x3 - x0);
  Real_t bisectY1 = Real_t(0.5) * (y2 + y1 - y3 - y0);
  Real_t bisectZ1 = Real_t(0.5) * (z2 + z1 - z3 - z0);
  Real_t areaX = Real_t(0.25) * (bisectY0 * bisectZ1 - bisectZ0 * bisectY1);
  Real_t areaY = Real_t(0.25) * (bisectZ0 * bisectX1 - bisectX0 * bisectZ1);
  Real_t areaZ = Real_t(0.25) * (bisectX0 * bisectY1 - bisectY0 * bisectX1);

  *normalX0 += areaX;
  *normalX1 += areaX;
  *normalX2 += areaX;
  *normalX3 += areaX;

  *normalY0 += areaY;
  *normalY1 += areaY;
  *normalY2 += areaY;
  *normalY3 += areaY;

  *normalZ0 += areaZ;
  *normalZ1 += areaZ;
  *normalZ2 += areaZ;
  *normalZ3 += areaZ;
}

__device__ inline void CalcElemNodeNormals(Real_t pfx[8], Real_t pfy[8],
                                           Real_t pfz[8], const Real_t x[8],
                                           const Real_t y[8], const Real_t z[8])
{
  for (Index_t i = 0; i < 8; ++i) {
    pfx[i] = Real_t(0.0);
    pfy[i] = Real_t(0.0);
    pfz[i] = Real_t(0.0);
  }
  SumElemFaceNormal(&pfx[0], &pfy[0], &pfz[0], &pfx[1], &pfy[1], &pfz[1],
                    &pfx[2], &pfy[2], &pfz[2], &pfx[3], &pfy[3], &pfz[3],
                    x[0], y[0], z[0], x[1], y[1], z[1], x[2], y[2], z[2], x[3],
                    y[3], z[3]);
  SumElemFaceNormal(&pfx[0], &pfy[0], &pfz[0], &pfx[4], &pfy[4], &pfz[4],
                    &pfx[5], &pfy[5], &pfz[5], &pfx[1], &pfy[1], &pfz[1],
                    x[0], y[0], z[0], x[4], y[4], z[4], x[5], y[5], z[5], x[1],
                    y[1], z[1]);
  SumElemFaceNormal(&pfx[1], &pfy[1], &pfz[1], &pfx[5], &pfy[5], &pfz[5],
                    &pfx[6], &pfy[6], &pfz[6], &pfx[2], &pfy[2], &pfz[2],
                    x[1], y[1], z[1], x[5], y[5], z[5], x[6], y[6], z[6], x[2],
                    y[2], z[2]);
  SumElemFaceNormal(&pfx[2], &pfy[2], &pfz[2], &pfx[6], &pfy[6], &pfz[6],
                    &pfx[7], &pfy[7], &pfz[7], &pfx[3], &pfy[3], &pfz[3],
                    x[2], y[2], z[2], x[6], y[6], z[6], x[7], y[7], z[7], x[3],
                    y[3], z[3]);
  SumElemFaceNormal(&pfx[3], &pfy[3], &pfz[3], &pfx[7], &pfy[7], &pfz[7],
                    &pfx[4], &pfy[4], &pfz[4], &pfx[0], &pfy[0], &pfz[0],
                    x[3], y[3], z[3], x[7], y[7], z[7], x[4], y[4], z[4], x[0],
                    y[0], z[0]);
  SumElemFaceNormal(&pfx[4], &pfy[4], &pfz[4], &pfx[7], &pfy[7], &pfz[7],
                    &pfx[6], &pfy[6], &pfz[6], &pfx[5], &pfy[5], &pfz[5],
                    x[4], y[4], z[4], x[7], y[7], z[7], x[6], y[6], z[6], x[5],
                    y[5], z[5]);
}

__device__ inline void SumElemStressesToNodeForces(const Real_t B[3][8],
                                                   const Real_t stress_xx,
                                                   const Real_t stress_yy,
                                                   const Real_t stress_zz,
                                                   Real_t fx[8], Real_t fy[8],
                                                   Real_t fz[8])
{
  for (Index_t i = 0; i < 8; i++) {
    fx[i] = -(stress_xx * B[0][i]);
    fy[i] = -(stress_yy * B[1][i]);
    fz[i] = -(stress_zz * B[2][i]);
  }
}

__device__ inline Real_t AreaFace(const Real_t x0, const Real_t x1,
                                  const Real_t x2, const Real_t x3,
                                  const Real_t y0, const Real_t y1,
                                  const Real_t y2, const Real_t y3,
                                  const Real_t z0, const Real_t z1,
                                  const Real_t z2, const Real_t z3)
{
  Real_t fx = (x2 - x0) - (x3 - x1);
  Real_t fy = (y2 - y0) - (y3 - y1);
  Real_t fz = (z2 - z0) - (z3 - z1);
  Real_t gx = (x2 - x0) + (x3 - x1);
  Real_t gy = (y2 - y0) + (y3 - y1);
  Real_t gz = (z2 - z0) + (z3 - z1);
  Real_t area = (fx * fx + fy * fy + fz * fz) * (gx * gx + gy * gy + gz * gz) -
                (fx * gx + fy * gy + fz * gz) * (fx * gx + fy * gy + fz * gz);
  return area;
}

__device__ inline Real_t CalcElemCharacteristicLength(const Real_t x[8],
                                                      const Real_t y[8],
                                                      const Real_t z[8],
                                                      const Real_t volume)
{
  Real_t a = AreaFace(x[0], x[1], x[2], x[3], y[0], y[1], y[2], y[3], z[0],
                      z[1], z[2], z[3]);
  Real_t charLength = a;

  a = AreaFace(x[4], x[5], x[6], x[7], y[4], y[5], y[6], y[7], z[4], z[5],
               z[6], z[7]);
  charLength = DeviceMax(a, charLength);

  a = AreaFace(x[0], x[1], x[5], x[4], y[0], y[1], y[5], y[4], z[0], z[1],
               z[5], z[4]);
  charLength = DeviceMax(a, charLength);

  a = AreaFace(x[1], x[2], x[6], x[5], y[1], y[2], y[6], y[5], z[1], z[2],
               z[6], z[5]);
  charLength = DeviceMax(a, charLength);

  a = AreaFace(x[2], x[3], x[7], x[6], y[2], y[3], y[7], y[6], z[2], z[3],
               z[7], z[6]);
  charLength = DeviceMax(a, charLength);

  a = AreaFace(x[3], x[0], x[4], x[7], y[3], y[0], y[4], y[7], z[3], z[0],
               z[4], z[7]);
  charLength = DeviceMax(a, charLength);

  charLength = Real_t(4.0) * volume / sqrt(charLength);

  return charLength;
}

__device__ inline void CalcElemVelocityGradient(const Real_t* xvel,
                                                const Real_t* yvel,
                                                const Real_t* zvel,
                                                const Real_t b[3][8],
                                                const Real_t detJ,
                                                Real_t* d)
{
  const Real_t inv_detJ = Real_t(1.0) / detJ;
  const Real_t* pfx = b[0];
  const Real_t* pfy = b[1];
  const Real_t* pfz = b[2];

  d[0] = inv_detJ *
         (pfx[0] * (xvel[0] - xvel[6]) + pfx[1] * (xvel[1] - xvel[7]) +
          pfx[2] * (xvel[2] - xvel[4]) + pfx[3] * (xvel[3] - xvel[5]));

  d[1] = inv_detJ *
         (pfy[0] * (yvel[0] - yvel[6]) + pfy[1] * (yvel[1] - yvel[7]) +
          pfy[2] * (yvel[2] - yvel[4]) + pfy[3] * (yvel[3] - yvel[5]));

  d[2] = inv_detJ *
         (pfz[0] * (zvel[0] - zvel[6]) + pfz[1] * (zvel[1] - zvel[7]) +
          pfz[2] * (zvel[2] - zvel[4]) + pfz[3] * (zvel[3] - zvel[5]));

  Real_t dyddx = inv_detJ *
                 (pfx[0] * (yvel[0] - yvel[6]) +
                  pfx[1] * (yvel[1] - yvel[7]) +
                  pfx[2] * (yvel[2] - yvel[4]) +
                  pfx[3] * (yvel[3] - yvel[5]));

  Real_t dxddy = inv_detJ *
                 (pfy[0] * (xvel[0] - xvel[6]) +
                  pfy[1] * (xvel[1] - xvel[7]) +
                  pfy[2] * (xvel[2] - xvel[4]) +
                  pfy[3] * (xvel[3] - xvel[5]));

  Real_t dzddx = inv_detJ *
                 (pfx[0] * (zvel[0] - zvel[6]) +
                  pfx[1] * (zvel[1] - zvel[7]) +
                  pfx[2] * (zvel[2] - zvel[4]) +
                  pfx[3] * (zvel[3] - zvel[5]));

  Real_t dxddz = inv_detJ *
                 (pfz[0] * (xvel[0] - xvel[6]) +
                  pfz[1] * (xvel[1] - xvel[7]) +
                  pfz[2] * (xvel[2] - xvel[4]) +
                  pfz[3] * (xvel[3] - xvel[5]));

  Real_t dzddy = inv_detJ *
                 (pfy[0] * (zvel[0] - zvel[6]) +
                  pfy[1] * (zvel[1] - zvel[7]) +
                  pfy[2] * (zvel[2] - zvel[4]) +
                  pfy[3] * (zvel[3] - zvel[5]));

  Real_t dyddz = inv_detJ *
                 (pfz[0] * (yvel[0] - yvel[6]) +
                  pfz[1] * (yvel[1] - yvel[7]) +
                  pfz[2] * (yvel[2] - yvel[4]) +
                  pfz[3] * (yvel[3] - yvel[5]));

  d[5] = Real_t(0.5) * (dxddy + dyddx);
  d[4] = Real_t(0.5) * (dxddz + dzddx);
  d[3] = Real_t(0.5) * (dzddy + dyddz);
}

__device__ inline void VoluDer(const Real_t x0, const Real_t x1,
                               const Real_t x2, const Real_t x3,
                               const Real_t x4, const Real_t x5,
                               const Real_t y0, const Real_t y1,
                               const Real_t y2, const Real_t y3,
                               const Real_t y4, const Real_t y5,
                               const Real_t z0, const Real_t z1,
                               const Real_t z2, const Real_t z3,
                               const Real_t z4, const Real_t z5,
                               Real_t* dvdx, Real_t* dvdy, Real_t* dvdz)
{
  const Real_t twelfth = Real_t(1.0) / Real_t(12.0);

  *dvdx = (y1 + y2) * (z0 + z1) - (y0 + y1) * (z1 + z2) +
          (y0 + y4) * (z3 + z4) - (y3 + y4) * (z0 + z4) -
          (y2 + y5) * (z3 + z5) + (y3 + y5) * (z2 + z5);
  *dvdy = -(x1 + x2) * (z0 + z1) + (x0 + x1) * (z1 + z2) -
          (x0 + x4) * (z3 + z4) + (x3 + x4) * (z0 + z4) +
          (x2 + x5) * (z3 + z5) - (x3 + x5) * (z2 + z5);

  *dvdz = -(y1 + y2) * (x0 + x1) + (y0 + y1) * (x1 + x2) -
          (y0 + y4) * (x3 + x4) + (y3 + y4) * (x0 + x4) +
          (y2 + y5) * (x3 + x5) - (y3 + y5) * (x2 + x5);

  *dvdx *= twelfth;
  *dvdy *= twelfth;
  *dvdz *= twelfth;
}

__device__ inline void CalcElemVolumeDerivative(Real_t dvdx[8], Real_t dvdy[8],
                                                Real_t dvdz[8],
                                                const Real_t x[8],
                                                const Real_t y[8],
                                                const Real_t z[8])
{
  VoluDer(x[1], x[2], x[3], x[4], x[5], x[7], y[1], y[2], y[3], y[4], y[5],
          y[7], z[1], z[2], z[3], z[4], z[5], z[7], &dvdx[0], &dvdy[0],
          &dvdz[0]);
  VoluDer(x[0], x[1], x[2], x[7], x[4], x[6], y[0], y[1], y[2], y[7], y[4],
          y[6], z[0], z[1], z[2], z[7], z[4], z[6], &dvdx[3], &dvdy[3],
          &dvdz[3]);
  VoluDer(x[3], x[0], x[1], x[6], x[7], x[5], y[3], y[0], y[1], y[6], y[7],
          y[5], z[3], z[0], z[1], z[6], z[7], z[5], &dvdx[2], &dvdy[2],
          &dvdz[2]);
  VoluDer(x[2], x[3], x[0], x[5], x[6], x[4], y[2], y[3], y[0], y[5], y[6],
          y[4], z[2], z[3], z[0], z[5], z[6], z[4], &dvdx[1], &dvdy[1],
          &dvdz[1]);
  VoluDer(x[7], x[6], x[5], x[0], x[3], x[1], y[7], y[6], y[5], y[0], y[3],
          y[1], z[7], z[6], z[5], z[0], z[3], z[1], &dvdx[4], &dvdy[4],
          &dvdz[4]);
  VoluDer(x[4], x[7], x[6], x[1], x[0], x[2], y[4], y[7], y[6], y[1], y[0],
          y[2], z[4], z[7], z[6], z[1], z[0], z[2], &dvdx[5], &dvdy[5],
          &dvdz[5]);
  VoluDer(x[5], x[4], x[7], x[2], x[1], x[3], y[5], y[4], y[7], y[2], y[1],
          y[3], z[5], z[4], z[7], z[2], z[1], z[3], &dvdx[6], &dvdy[6],
          &dvdz[6]);
  VoluDer(x[6], x[5], x[4], x[3], x[2], x[0], y[6], y[5], y[4], y[3], y[2],
          y[0], z[6], z[5], z[4], z[3], z[2], z[0], &dvdx[7], &dvdy[7],
          &dvdz[7]);
}

__device__ inline void CalcElemFBHourglassForce(Real_t* xd, Real_t* yd,
                                                Real_t* zd,
                                                Real_t hourgam[8][4],
                                                Real_t coefficient,
                                                Real_t* hgfx, Real_t* hgfy,
                                                Real_t* hgfz)
{
  Real_t hxx[4];
  for (Index_t i = 0; i < 4; i++) {
    hxx[i] = hourgam[0][i] * xd[0] + hourgam[1][i] * xd[1] +
             hourgam[2][i] * xd[2] + hourgam[3][i] * xd[3] +
             hourgam[4][i] * xd[4] + hourgam[5][i] * xd[5] +
             hourgam[6][i] * xd[6] + hourgam[7][i] * xd[7];
  }
  for (Index_t i = 0; i < 8; i++) {
    hgfx[i] = coefficient *
              (hourgam[i][0] * hxx[0] + hourgam[i][1] * hxx[1] +
               hourgam[i][2] * hxx[2] + hourgam[i][3] * hxx[3]);
  }
  for (Index_t i = 0; i < 4; i++) {
    hxx[i] = hourgam[0][i] * yd[0] + hourgam[1][i] * yd[1] +
             hourgam[2][i] * yd[2] + hourgam[3][i] * yd[3] +
             hourgam[4][i] * yd[4] + hourgam[5][i] * yd[5] +
             hourgam[6][i] * yd[6] + hourgam[7][i] * yd[7];
  }
  for (Index_t i = 0; i < 8; i++) {
    hgfy[i] = coefficient *
              (hourgam[i][0] * hxx[0] + hourgam[i][1] * hxx[1] +
               hourgam[i][2] * hxx[2] + hourgam[i][3] * hxx[3]);
  }
  for (Index_t i = 0; i < 4; i++) {
    hxx[i] = hourgam[0][i] * zd[0] + hourgam[1][i] * zd[1] +
             hourgam[2][i] * zd[2] + hourgam[3][i] * zd[3] +
             hourgam[4][i] * zd[4] + hourgam[5][i] * zd[5] +
             hourgam[6][i] * zd[6] + hourgam[7][i] * zd[7];
  }
  for (Index_t i = 0; i < 8; i++) {
    hgfz[i] = coefficient *
              (hourgam[i][0] * hxx[0] + hourgam[i][1] * hxx[1] +
               hourgam[i][2] * hxx[2] + hourgam[i][3] * hxx[3]);
  }
}

struct DeviceDomain {
  Index_t numElem;
  Index_t numNode;
  Real_t* x;
  Real_t* y;
  Real_t* z;
  Real_t* xd;
  Real_t* yd;
  Real_t* zd;
  Real_t* xdd;
  Real_t* ydd;
  Real_t* zdd;
  Real_t* fx;
  Real_t* fy;
  Real_t* fz;
  Real_t* nodalMass;

  Real_t* e;
  Real_t* p;
  Real_t* q;
  Real_t* ql;
  Real_t* qq;
  Real_t* v;
  Real_t* vnew;
  Real_t* delv;
  Real_t* vdov;
  Real_t* arealg;
  Real_t* volo;
  Real_t* ss;
  Real_t* elemMass;

  Index_t* nodelist;
  Index_t* lxim;
  Index_t* lxip;
  Index_t* letam;
  Index_t* letap;
  Index_t* lzetam;
  Index_t* lzetap;
  Int_t* elemBC;

  Index_t* symmX;
  Index_t* symmY;
  Index_t* symmZ;
  Index_t numSymmX;
  Index_t numSymmY;
  Index_t numSymmZ;

  Index_t* nodeElemStart;
  Index_t* nodeElemCornerList;

  Real_t* dxx;
  Real_t* dyy;
  Real_t* dzz;
  Real_t* delv_xi;
  Real_t* delv_eta;
  Real_t* delv_zeta;
  Real_t* delx_xi;
  Real_t* delx_eta;
  Real_t* delx_zeta;

  Real_t* determ;
  Real_t* dvdx;
  Real_t* dvdy;
  Real_t* dvdz;
  Real_t* x8n;
  Real_t* y8n;
  Real_t* z8n;
  Real_t* fx_elem;
  Real_t* fy_elem;
  Real_t* fz_elem;

  Real_t* dtcourant_per_elem;
  Real_t* dthydro_per_elem;

  int* error;
};

struct GpuContext {
  DeviceDomain d;
  std::vector<Index_t> nodeElemStart;
  std::vector<Index_t> nodeElemCornerList;
  std::vector<Index_t> symmX;
  std::vector<Index_t> symmY;
  std::vector<Index_t> symmZ;
};

__device__ inline void SetDeviceError(int* error, int code)
{
  atomicCAS(error, 0, code);
}

__global__ void KernelZeroForces(Index_t numNode, Real_t* fx, Real_t* fy,
                                 Real_t* fz)
{
  Index_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < numNode) {
    fx[i] = Real_t(0.0);
    fy[i] = Real_t(0.0);
    fz[i] = Real_t(0.0);
  }
}

__global__ void KernelIntegrateStress(Index_t numElem, const Index_t* nodelist,
                                      const Real_t* x, const Real_t* y,
                                      const Real_t* z, const Real_t* p,
                                      const Real_t* q, Real_t* determ,
                                      Real_t* fx_elem, Real_t* fy_elem,
                                      Real_t* fz_elem, int* error)
{
  Index_t k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < numElem) {
    Real_t B[3][8];
    Real_t x_local[8];
    Real_t y_local[8];
    Real_t z_local[8];
    Real_t fx_local[8];
    Real_t fy_local[8];
    Real_t fz_local[8];

    const Index_t* elemToNode = &nodelist[k * 8];
    for (Index_t lnode = 0; lnode < 8; ++lnode) {
      Index_t gnode = elemToNode[lnode];
      x_local[lnode] = x[gnode];
      y_local[lnode] = y[gnode];
      z_local[lnode] = z[gnode];
    }

    Real_t detJ;
    CalcElemShapeFunctionDerivatives(x_local, y_local, z_local, B, &detJ);
    CalcElemNodeNormals(B[0], B[1], B[2], x_local, y_local, z_local);

    Real_t sig = -p[k] - q[k];
    SumElemStressesToNodeForces(B, sig, sig, sig, fx_local, fy_local, fz_local);

    determ[k] = detJ;
    if (detJ <= Real_t(0.0)) {
      SetDeviceError(error, VolumeError);
    }

    Index_t base = k * 8;
    for (Index_t i = 0; i < 8; ++i) {
      fx_elem[base + i] = fx_local[i];
      fy_elem[base + i] = fy_local[i];
      fz_elem[base + i] = fz_local[i];
    }
  }
}

__global__ void KernelGatherForces(Index_t numNode, const Index_t* nodeElemStart,
                                   const Index_t* nodeElemCornerList,
                                   const Real_t* fx_elem,
                                   const Real_t* fy_elem,
                                   const Real_t* fz_elem, Real_t* fx,
                                   Real_t* fy, Real_t* fz, int add)
{
  Index_t gnode = blockIdx.x * blockDim.x + threadIdx.x;
  if (gnode < numNode) {
    Real_t fx_sum = Real_t(0.0);
    Real_t fy_sum = Real_t(0.0);
    Real_t fz_sum = Real_t(0.0);
    Index_t start = nodeElemStart[gnode];
    Index_t end = nodeElemStart[gnode + 1];
    for (Index_t i = start; i < end; ++i) {
      Index_t corner = nodeElemCornerList[i];
      fx_sum += fx_elem[corner];
      fy_sum += fy_elem[corner];
      fz_sum += fz_elem[corner];
    }
    if (add) {
      fx[gnode] += fx_sum;
      fy[gnode] += fy_sum;
      fz[gnode] += fz_sum;
    } else {
      fx[gnode] = fx_sum;
      fy[gnode] = fy_sum;
      fz[gnode] = fz_sum;
    }
  }
}

__global__ void KernelCalcHourglassInputs(Index_t numElem,
                                          const Index_t* nodelist,
                                          const Real_t* x, const Real_t* y,
                                          const Real_t* z, const Real_t* v,
                                          const Real_t* volo, Real_t* determ,
                                          Real_t* dvdx, Real_t* dvdy,
                                          Real_t* dvdz, Real_t* x8n,
                                          Real_t* y8n, Real_t* z8n,
                                          int* error)
{
  Index_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < numElem) {
    Real_t x1[8];
    Real_t y1[8];
    Real_t z1[8];
    Real_t pfx[8];
    Real_t pfy[8];
    Real_t pfz[8];

    const Index_t* elemToNode = &nodelist[i * 8];
    for (Index_t ii = 0; ii < 8; ++ii) {
      Index_t gnode = elemToNode[ii];
      x1[ii] = x[gnode];
      y1[ii] = y[gnode];
      z1[ii] = z[gnode];
    }

    CalcElemVolumeDerivative(pfx, pfy, pfz, x1, y1, z1);

    Index_t base = i * 8;
    for (Index_t ii = 0; ii < 8; ++ii) {
      Index_t jj = base + ii;
      dvdx[jj] = pfx[ii];
      dvdy[jj] = pfy[ii];
      dvdz[jj] = pfz[ii];
      x8n[jj] = x1[ii];
      y8n[jj] = y1[ii];
      z8n[jj] = z1[ii];
    }

    determ[i] = volo[i] * v[i];
    if (v[i] <= Real_t(0.0)) {
      SetDeviceError(error, VolumeError);
    }
  }
}

__global__ void KernelCalcHourglassForces(Index_t numElem,
                                          const Index_t* nodelist,
                                          const Real_t* xd, const Real_t* yd,
                                          const Real_t* zd, const Real_t* ss,
                                          const Real_t* elemMass,
                                          const Real_t* determ,
                                          const Real_t* dvdx,
                                          const Real_t* dvdy,
                                          const Real_t* dvdz,
                                          const Real_t* x8n,
                                          const Real_t* y8n,
                                          const Real_t* z8n, Real_t hourg,
                                          Real_t* fx_elem, Real_t* fy_elem,
                                          Real_t* fz_elem)
{
  Index_t i2 = blockIdx.x * blockDim.x + threadIdx.x;
  if (i2 < numElem) {
    Real_t hgfx[8];
    Real_t hgfy[8];
    Real_t hgfz[8];
    Real_t hourgam[8][4];
    Real_t xd1[8];
    Real_t yd1[8];
    Real_t zd1[8];

    const Real_t gamma[4][8] = {
        {Real_t(1.), Real_t(1.), Real_t(-1.), Real_t(-1.), Real_t(-1.),
         Real_t(-1.), Real_t(1.), Real_t(1.)},
        {Real_t(1.), Real_t(-1.), Real_t(-1.), Real_t(1.), Real_t(-1.),
         Real_t(1.), Real_t(1.), Real_t(-1.)},
        {Real_t(1.), Real_t(-1.), Real_t(1.), Real_t(-1.), Real_t(1.),
         Real_t(-1.), Real_t(1.), Real_t(-1.)},
        {Real_t(-1.), Real_t(1.), Real_t(-1.), Real_t(1.), Real_t(1.),
         Real_t(-1.), Real_t(1.), Real_t(-1.)}};

    Index_t base = i2 * 8;
    Real_t volinv = Real_t(1.0) / determ[i2];

    for (Index_t i1 = 0; i1 < 4; ++i1) {
      Real_t hourmodx = x8n[base] * gamma[i1][0] + x8n[base + 1] * gamma[i1][1] +
                        x8n[base + 2] * gamma[i1][2] +
                        x8n[base + 3] * gamma[i1][3] +
                        x8n[base + 4] * gamma[i1][4] +
                        x8n[base + 5] * gamma[i1][5] +
                        x8n[base + 6] * gamma[i1][6] +
                        x8n[base + 7] * gamma[i1][7];

      Real_t hourmody = y8n[base] * gamma[i1][0] + y8n[base + 1] * gamma[i1][1] +
                        y8n[base + 2] * gamma[i1][2] +
                        y8n[base + 3] * gamma[i1][3] +
                        y8n[base + 4] * gamma[i1][4] +
                        y8n[base + 5] * gamma[i1][5] +
                        y8n[base + 6] * gamma[i1][6] +
                        y8n[base + 7] * gamma[i1][7];

      Real_t hourmodz = z8n[base] * gamma[i1][0] + z8n[base + 1] * gamma[i1][1] +
                        z8n[base + 2] * gamma[i1][2] +
                        z8n[base + 3] * gamma[i1][3] +
                        z8n[base + 4] * gamma[i1][4] +
                        z8n[base + 5] * gamma[i1][5] +
                        z8n[base + 6] * gamma[i1][6] +
                        z8n[base + 7] * gamma[i1][7];

      for (Index_t n = 0; n < 8; ++n) {
        hourgam[n][i1] =
            gamma[i1][n] - volinv * (dvdx[base + n] * hourmodx +
                                     dvdy[base + n] * hourmody +
                                     dvdz[base + n] * hourmodz);
      }
    }

    const Index_t* elemToNode = &nodelist[base];
    for (Index_t i = 0; i < 8; ++i) {
      Index_t n = elemToNode[i];
      xd1[i] = xd[n];
      yd1[i] = yd[n];
      zd1[i] = zd[n];
    }

    Real_t ss1 = ss[i2];
    Real_t mass1 = elemMass[i2];
    Real_t volume13 = cbrt(determ[i2]);
    Real_t coefficient = -hourg * Real_t(0.01) * ss1 * mass1 / volume13;

    CalcElemFBHourglassForce(xd1, yd1, zd1, hourgam, coefficient, hgfx, hgfy,
                             hgfz);

    for (Index_t i = 0; i < 8; ++i) {
      fx_elem[base + i] = hgfx[i];
      fy_elem[base + i] = hgfy[i];
      fz_elem[base + i] = hgfz[i];
    }
  }
}

__global__ void KernelCalcAcceleration(Index_t numNode, const Real_t* fx,
                                       const Real_t* fy, const Real_t* fz,
                                       const Real_t* nodalMass, Real_t* xdd,
                                       Real_t* ydd, Real_t* zdd)
{
  Index_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < numNode) {
    xdd[i] = fx[i] / nodalMass[i];
    ydd[i] = fy[i] / nodalMass[i];
    zdd[i] = fz[i] / nodalMass[i];
  }
}

__global__ void KernelApplyAccelBC(Index_t count, const Index_t* symmNodes,
                                   Real_t* accel)
{
  Index_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    accel[symmNodes[i]] = Real_t(0.0);
  }
}

__global__ void KernelCalcVelocity(Index_t numNode, Real_t dt, Real_t u_cut,
                                   Real_t* xd, Real_t* yd, Real_t* zd,
                                   const Real_t* xdd, const Real_t* ydd,
                                   const Real_t* zdd)
{
  Index_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < numNode) {
    Real_t xdtmp = xd[i] + xdd[i] * dt;
    if (fabs(xdtmp) < u_cut) {
      xdtmp = Real_t(0.0);
    }
    xd[i] = xdtmp;

    Real_t ydtmp = yd[i] + ydd[i] * dt;
    if (fabs(ydtmp) < u_cut) {
      ydtmp = Real_t(0.0);
    }
    yd[i] = ydtmp;

    Real_t zdtmp = zd[i] + zdd[i] * dt;
    if (fabs(zdtmp) < u_cut) {
      zdtmp = Real_t(0.0);
    }
    zd[i] = zdtmp;
  }
}

__global__ void KernelCalcPosition(Index_t numNode, Real_t dt, Real_t* x,
                                   Real_t* y, Real_t* z, const Real_t* xd,
                                   const Real_t* yd, const Real_t* zd)
{
  Index_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < numNode) {
    x[i] += xd[i] * dt;
    y[i] += yd[i] * dt;
    z[i] += zd[i] * dt;
  }
}

__global__ void KernelCalcKinematics(Index_t numElem, const Index_t* nodelist,
                                     const Real_t* x, const Real_t* y,
                                     const Real_t* z, const Real_t* xd,
                                     const Real_t* yd, const Real_t* zd,
                                     const Real_t* volo, const Real_t* v,
                                     Real_t dt, Real_t* vnew, Real_t* delv,
                                     Real_t* arealg, Real_t* dxx,
                                     Real_t* dyy, Real_t* dzz)
{
  Index_t k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < numElem) {
    Real_t B[3][8];
    Real_t D[6];
    Real_t x_local[8];
    Real_t y_local[8];
    Real_t z_local[8];
    Real_t xd_local[8];
    Real_t yd_local[8];
    Real_t zd_local[8];

    const Index_t* elemToNode = &nodelist[k * 8];
    for (Index_t lnode = 0; lnode < 8; ++lnode) {
      Index_t gnode = elemToNode[lnode];
      x_local[lnode] = x[gnode];
      y_local[lnode] = y[gnode];
      z_local[lnode] = z[gnode];
      xd_local[lnode] = xd[gnode];
      yd_local[lnode] = yd[gnode];
      zd_local[lnode] = zd[gnode];
    }

    Real_t volume = CalcElemVolumeDevice(x_local, y_local, z_local);
    Real_t relativeVolume = volume / volo[k];
    vnew[k] = relativeVolume;
    delv[k] = relativeVolume - v[k];
    arealg[k] = CalcElemCharacteristicLength(x_local, y_local, z_local, volume);

    Real_t dt2 = Real_t(0.5) * dt;
    for (Index_t j = 0; j < 8; ++j) {
      x_local[j] -= dt2 * xd_local[j];
      y_local[j] -= dt2 * yd_local[j];
      z_local[j] -= dt2 * zd_local[j];
    }

    Real_t detJ = Real_t(0.0);
    CalcElemShapeFunctionDerivatives(x_local, y_local, z_local, B, &detJ);
    CalcElemVelocityGradient(xd_local, yd_local, zd_local, B, detJ, D);

    dxx[k] = D[0];
    dyy[k] = D[1];
    dzz[k] = D[2];
  }
}

__global__ void KernelCalcLagrangeElements(Index_t numElem, Real_t* dxx,
                                           Real_t* dyy, Real_t* dzz,
                                           const Real_t* vnew,
                                           Real_t* vdov, int* error)
{
  Index_t k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < numElem) {
    Real_t vdov_local = dxx[k] + dyy[k] + dzz[k];
    Real_t vdovthird = vdov_local / Real_t(3.0);

    vdov[k] = vdov_local;
    dxx[k] -= vdovthird;
    dyy[k] -= vdovthird;
    dzz[k] -= vdovthird;

    if (vnew[k] <= Real_t(0.0)) {
      SetDeviceError(error, VolumeError);
    }
  }
}

__global__ void KernelCalcMonotonicQGradients(Index_t numElem,
                                              const Index_t* nodelist,
                                              const Real_t* x,
                                              const Real_t* y,
                                              const Real_t* z,
                                              const Real_t* xd,
                                              const Real_t* yd,
                                              const Real_t* zd,
                                              const Real_t* volo,
                                              const Real_t* vnew,
                                              Real_t* delv_xi,
                                              Real_t* delv_eta,
                                              Real_t* delv_zeta,
                                              Real_t* delx_xi,
                                              Real_t* delx_eta,
                                              Real_t* delx_zeta)
{
  Index_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < numElem) {
    const Real_t ptiny = Real_t(1.e-36);
    Real_t ax, ay, az;
    Real_t dxv, dyv, dzv;

    const Index_t* elemToNode = &nodelist[i * 8];
    Index_t n0 = elemToNode[0];
    Index_t n1 = elemToNode[1];
    Index_t n2 = elemToNode[2];
    Index_t n3 = elemToNode[3];
    Index_t n4 = elemToNode[4];
    Index_t n5 = elemToNode[5];
    Index_t n6 = elemToNode[6];
    Index_t n7 = elemToNode[7];

    Real_t x0 = x[n0];
    Real_t x1 = x[n1];
    Real_t x2 = x[n2];
    Real_t x3 = x[n3];
    Real_t x4 = x[n4];
    Real_t x5 = x[n5];
    Real_t x6 = x[n6];
    Real_t x7 = x[n7];

    Real_t y0 = y[n0];
    Real_t y1 = y[n1];
    Real_t y2 = y[n2];
    Real_t y3 = y[n3];
    Real_t y4 = y[n4];
    Real_t y5 = y[n5];
    Real_t y6 = y[n6];
    Real_t y7 = y[n7];

    Real_t z0 = z[n0];
    Real_t z1 = z[n1];
    Real_t z2 = z[n2];
    Real_t z3 = z[n3];
    Real_t z4 = z[n4];
    Real_t z5 = z[n5];
    Real_t z6 = z[n6];
    Real_t z7 = z[n7];

    Real_t xv0 = xd[n0];
    Real_t xv1 = xd[n1];
    Real_t xv2 = xd[n2];
    Real_t xv3 = xd[n3];
    Real_t xv4 = xd[n4];
    Real_t xv5 = xd[n5];
    Real_t xv6 = xd[n6];
    Real_t xv7 = xd[n7];

    Real_t yv0 = yd[n0];
    Real_t yv1 = yd[n1];
    Real_t yv2 = yd[n2];
    Real_t yv3 = yd[n3];
    Real_t yv4 = yd[n4];
    Real_t yv5 = yd[n5];
    Real_t yv6 = yd[n6];
    Real_t yv7 = yd[n7];

    Real_t zv0 = zd[n0];
    Real_t zv1 = zd[n1];
    Real_t zv2 = zd[n2];
    Real_t zv3 = zd[n3];
    Real_t zv4 = zd[n4];
    Real_t zv5 = zd[n5];
    Real_t zv6 = zd[n6];
    Real_t zv7 = zd[n7];

    Real_t vol = volo[i] * vnew[i];
    Real_t norm = Real_t(1.0) / (vol + ptiny);

    Real_t dxj = Real_t(-0.25) * ((x0 + x1 + x5 + x4) - (x3 + x2 + x6 + x7));
    Real_t dyj = Real_t(-0.25) * ((y0 + y1 + y5 + y4) - (y3 + y2 + y6 + y7));
    Real_t dzj = Real_t(-0.25) * ((z0 + z1 + z5 + z4) - (z3 + z2 + z6 + z7));

    Real_t dxi = Real_t(0.25) * ((x1 + x2 + x6 + x5) - (x0 + x3 + x7 + x4));
    Real_t dyi = Real_t(0.25) * ((y1 + y2 + y6 + y5) - (y0 + y3 + y7 + y4));
    Real_t dzi = Real_t(0.25) * ((z1 + z2 + z6 + z5) - (z0 + z3 + z7 + z4));

    Real_t dxk = Real_t(0.25) * ((x4 + x5 + x6 + x7) - (x0 + x1 + x2 + x3));
    Real_t dyk = Real_t(0.25) * ((y4 + y5 + y6 + y7) - (y0 + y1 + y2 + y3));
    Real_t dzk = Real_t(0.25) * ((z4 + z5 + z6 + z7) - (z0 + z1 + z2 + z3));

    ax = dyi * dzj - dzi * dyj;
    ay = dzi * dxj - dxi * dzj;
    az = dxi * dyj - dyi * dxj;

    delx_zeta[i] = vol / sqrt(ax * ax + ay * ay + az * az + ptiny);

    ax *= norm;
    ay *= norm;
    az *= norm;

    dxv = Real_t(0.25) * ((xv4 + xv5 + xv6 + xv7) - (xv0 + xv1 + xv2 + xv3));
    dyv = Real_t(0.25) * ((yv4 + yv5 + yv6 + yv7) - (yv0 + yv1 + yv2 + yv3));
    dzv = Real_t(0.25) * ((zv4 + zv5 + zv6 + zv7) - (zv0 + zv1 + zv2 + zv3));

    delv_zeta[i] = ax * dxv + ay * dyv + az * dzv;

    ax = dyj * dzk - dzj * dyk;
    ay = dzj * dxk - dxj * dzk;
    az = dxj * dyk - dyj * dxk;

    delx_xi[i] = vol / sqrt(ax * ax + ay * ay + az * az + ptiny);

    ax *= norm;
    ay *= norm;
    az *= norm;

    dxv = Real_t(0.25) * ((xv1 + xv2 + xv6 + xv5) - (xv0 + xv3 + xv7 + xv4));
    dyv = Real_t(0.25) * ((yv1 + yv2 + yv6 + yv5) - (yv0 + yv3 + yv7 + yv4));
    dzv = Real_t(0.25) * ((zv1 + zv2 + zv6 + zv5) - (zv0 + zv3 + zv7 + zv4));

    delv_xi[i] = ax * dxv + ay * dyv + az * dzv;

    ax = dyk * dzi - dzk * dyi;
    ay = dzk * dxi - dxk * dzi;
    az = dxk * dyi - dyk * dxi;

    delx_eta[i] = vol / sqrt(ax * ax + ay * ay + az * az + ptiny);

    ax *= norm;
    ay *= norm;
    az *= norm;

    dxv = Real_t(-0.25) * ((xv0 + xv1 + xv5 + xv4) - (xv3 + xv2 + xv6 + xv7));
    dyv = Real_t(-0.25) * ((yv0 + yv1 + yv5 + yv4) - (yv3 + yv2 + yv6 + yv7));
    dzv = Real_t(-0.25) * ((zv0 + zv1 + zv5 + zv4) - (zv3 + zv2 + zv6 + zv7));

    delv_eta[i] = ax * dxv + ay * dyv + az * dzv;
  }
}

__global__ void KernelCalcMonotonicQ(Index_t numElem, const Int_t* elemBC,
                                     const Index_t* lxim,
                                     const Index_t* lxip,
                                     const Index_t* letam,
                                     const Index_t* letap,
                                     const Index_t* lzetam,
                                     const Index_t* lzetap,
                                     const Real_t* delv_xi,
                                     const Real_t* delv_eta,
                                     const Real_t* delv_zeta,
                                     const Real_t* delx_xi,
                                     const Real_t* delx_eta,
                                     const Real_t* delx_zeta,
                                     const Real_t* vdov,
                                     const Real_t* elemMass,
                                     const Real_t* volo, const Real_t* vnew,
                                     Real_t monoq_limiter_mult,
                                     Real_t monoq_max_slope,
                                     Real_t qlc_monoq, Real_t qqc_monoq,
                                     Real_t* qq, Real_t* ql)
{
  Index_t ielem = blockIdx.x * blockDim.x + threadIdx.x;
  if (ielem < numElem) {
    const Real_t ptiny = Real_t(1.e-36);
    Real_t qlin;
    Real_t qquad;
    Real_t phixi;
    Real_t phieta;
    Real_t phizeta;
    Int_t bcMask = elemBC[ielem];
    Real_t delvm = Real_t(0.0);
    Real_t delvp = Real_t(0.0);

    Real_t norm = Real_t(1.0) / (delv_xi[ielem] + ptiny);

    switch (bcMask & XI_M) {
      case XI_M_COMM:
      case 0:
        delvm = delv_xi[lxim[ielem]];
        break;
      case XI_M_SYMM:
        delvm = delv_xi[ielem];
        break;
      case XI_M_FREE:
        delvm = Real_t(0.0);
        break;
      default:
        delvm = Real_t(0.0);
        break;
    }
    switch (bcMask & XI_P) {
      case XI_P_COMM:
      case 0:
        delvp = delv_xi[lxip[ielem]];
        break;
      case XI_P_SYMM:
        delvp = delv_xi[ielem];
        break;
      case XI_P_FREE:
        delvp = Real_t(0.0);
        break;
      default:
        delvp = Real_t(0.0);
        break;
    }

    delvm = delvm * norm;
    delvp = delvp * norm;

    phixi = Real_t(0.5) * (delvm + delvp);

    delvm *= monoq_limiter_mult;
    delvp *= monoq_limiter_mult;

    if (delvm < phixi) {
      phixi = delvm;
    }
    if (delvp < phixi) {
      phixi = delvp;
    }
    if (phixi < Real_t(0.0)) {
      phixi = Real_t(0.0);
    }
    if (phixi > monoq_max_slope) {
      phixi = monoq_max_slope;
    }

    norm = Real_t(1.0) / (delv_eta[ielem] + ptiny);

    switch (bcMask & ETA_M) {
      case ETA_M_COMM:
      case 0:
        delvm = delv_eta[letam[ielem]];
        break;
      case ETA_M_SYMM:
        delvm = delv_eta[ielem];
        break;
      case ETA_M_FREE:
        delvm = Real_t(0.0);
        break;
      default:
        delvm = Real_t(0.0);
        break;
    }
    switch (bcMask & ETA_P) {
      case ETA_P_COMM:
      case 0:
        delvp = delv_eta[letap[ielem]];
        break;
      case ETA_P_SYMM:
        delvp = delv_eta[ielem];
        break;
      case ETA_P_FREE:
        delvp = Real_t(0.0);
        break;
      default:
        delvp = Real_t(0.0);
        break;
    }

    delvm = delvm * norm;
    delvp = delvp * norm;

    phieta = Real_t(0.5) * (delvm + delvp);

    delvm *= monoq_limiter_mult;
    delvp *= monoq_limiter_mult;

    if (delvm < phieta) {
      phieta = delvm;
    }
    if (delvp < phieta) {
      phieta = delvp;
    }
    if (phieta < Real_t(0.0)) {
      phieta = Real_t(0.0);
    }
    if (phieta > monoq_max_slope) {
      phieta = monoq_max_slope;
    }

    norm = Real_t(1.0) / (delv_zeta[ielem] + ptiny);

    switch (bcMask & ZETA_M) {
      case ZETA_M_COMM:
      case 0:
        delvm = delv_zeta[lzetam[ielem]];
        break;
      case ZETA_M_SYMM:
        delvm = delv_zeta[ielem];
        break;
      case ZETA_M_FREE:
        delvm = Real_t(0.0);
        break;
      default:
        delvm = Real_t(0.0);
        break;
    }
    switch (bcMask & ZETA_P) {
      case ZETA_P_COMM:
      case 0:
        delvp = delv_zeta[lzetap[ielem]];
        break;
      case ZETA_P_SYMM:
        delvp = delv_zeta[ielem];
        break;
      case ZETA_P_FREE:
        delvp = Real_t(0.0);
        break;
      default:
        delvp = Real_t(0.0);
        break;
    }

    delvm = delvm * norm;
    delvp = delvp * norm;

    phizeta = Real_t(0.5) * (delvm + delvp);

    delvm *= monoq_limiter_mult;
    delvp *= monoq_limiter_mult;

    if (delvm < phizeta) {
      phizeta = delvm;
    }
    if (delvp < phizeta) {
      phizeta = delvp;
    }
    if (phizeta < Real_t(0.0)) {
      phizeta = Real_t(0.0);
    }
    if (phizeta > monoq_max_slope) {
      phizeta = monoq_max_slope;
    }

    if (vdov[ielem] > Real_t(0.0)) {
      qlin = Real_t(0.0);
      qquad = Real_t(0.0);
    } else {
      Real_t delvxxi = delv_xi[ielem] * delx_xi[ielem];
      Real_t delvxeta = delv_eta[ielem] * delx_eta[ielem];
      Real_t delvxzeta = delv_zeta[ielem] * delx_zeta[ielem];

      if (delvxxi > Real_t(0.0)) {
        delvxxi = Real_t(0.0);
      }
      if (delvxeta > Real_t(0.0)) {
        delvxeta = Real_t(0.0);
      }
      if (delvxzeta > Real_t(0.0)) {
        delvxzeta = Real_t(0.0);
      }

      Real_t rho = elemMass[ielem] / (volo[ielem] * vnew[ielem]);

      qlin = -qlc_monoq * rho *
             (delvxxi * (Real_t(1.0) - phixi) +
              delvxeta * (Real_t(1.0) - phieta) +
              delvxzeta * (Real_t(1.0) - phizeta));

      qquad = qqc_monoq * rho *
              (delvxxi * delvxxi * (Real_t(1.0) - phixi * phixi) +
               delvxeta * delvxeta * (Real_t(1.0) - phieta * phieta) +
               delvxzeta * delvxzeta * (Real_t(1.0) - phizeta * phizeta));
    }

    qq[ielem] = qquad;
    ql[ielem] = qlin;
  }
}

__global__ void KernelCheckQStop(Index_t numElem, const Real_t* q,
                                 Real_t qstop, int* error)
{
  Index_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < numElem) {
    if (q[i] > qstop) {
      SetDeviceError(error, QStopError);
    }
  }
}

__global__ void KernelEvalEOS(Index_t numElem, const Real_t* vnew,
                              const Real_t* delv, const Real_t* ql,
                              const Real_t* qq, const Real_t* v,
                              const Real_t* volo, const Real_t* elemMass,
                              Real_t e_cut, Real_t p_cut, Real_t q_cut,
                              Real_t pmin, Real_t emin, Real_t eosvmax,
                              Real_t eosvmin, Real_t rho0, Real_t ss4o3,
                              Real_t* e, Real_t* p, Real_t* q, Real_t* ss,
                              int* error)
{
  Index_t ielem = blockIdx.x * blockDim.x + threadIdx.x;
  if (ielem < numElem) {
    Real_t vnewc = vnew[ielem];
    if (eosvmin != Real_t(0.0) && vnewc < eosvmin) {
      vnewc = eosvmin;
    }
    if (eosvmax != Real_t(0.0) && vnewc > eosvmax) {
      vnewc = eosvmax;
    }

    Real_t vc = v[ielem];
    if (eosvmin != Real_t(0.0) && vc < eosvmin) {
      vc = eosvmin;
    }
    if (eosvmax != Real_t(0.0) && vc > eosvmax) {
      vc = eosvmax;
    }
    if (vc <= Real_t(0.0)) {
      SetDeviceError(error, VolumeError);
    }

    Real_t delvc = delv[ielem];
    Real_t e_old = e[ielem];
    Real_t p_old = p[ielem];
    Real_t q_old = q[ielem];
    Real_t qq_old = qq[ielem];
    Real_t ql_old = ql[ielem];

    Real_t compression = Real_t(1.0) / vnewc - Real_t(1.0);
    Real_t vchalf = vnewc - delvc * Real_t(0.5);
    Real_t compHalfStep = Real_t(1.0) / vchalf - Real_t(1.0);

    if (eosvmin != Real_t(0.0) && vnewc <= eosvmin) {
      compHalfStep = compression;
    }

    if (eosvmax != Real_t(0.0) && vnewc >= eosvmax) {
      p_old = Real_t(0.0);
      compression = Real_t(0.0);
      compHalfStep = Real_t(0.0);
    }

    Real_t work = Real_t(0.0);

    Real_t e_new = e_old - Real_t(0.5) * delvc * (p_old + q_old) +
                   Real_t(0.5) * work;
    if (e_new < emin) {
      e_new = emin;
    }

    Real_t c1s = Real_t(2.0) / Real_t(3.0);
    Real_t bvc = c1s * (compression + Real_t(1.0));
    Real_t pbvc = c1s;

    Real_t pHalfStep = bvc * e_new;
    if (fabs(pHalfStep) < p_cut) {
      pHalfStep = Real_t(0.0);
    }
    if (vnewc >= eosvmax) {
      pHalfStep = Real_t(0.0);
    }
    if (pHalfStep < pmin) {
      pHalfStep = pmin;
    }

    Real_t q_new;
    if (delvc > Real_t(0.0)) {
      q_new = Real_t(0.0);
    } else {
      Real_t vhalf = Real_t(1.0) / (Real_t(1.0) + compHalfStep);
      Real_t ssc = (pbvc * e_new + vhalf * vhalf * bvc * pHalfStep) / rho0;
      if (ssc <= Real_t(.1111111e-36)) {
        ssc = Real_t(.3333333e-18);
      } else {
        ssc = sqrt(ssc);
      }
      q_new = (ssc * ql_old + qq_old);
    }

    e_new = e_new + Real_t(0.5) * delvc *
                        (Real_t(3.0) * (p_old + q_old) -
                         Real_t(4.0) * (pHalfStep + q_new));

    e_new += Real_t(0.5) * work;
    if (fabs(e_new) < e_cut) {
      e_new = Real_t(0.0);
    }
    if (e_new < emin) {
      e_new = emin;
    }

    Real_t p_new = bvc * e_new;
    if (fabs(p_new) < p_cut) {
      p_new = Real_t(0.0);
    }
    if (vnewc >= eosvmax) {
      p_new = Real_t(0.0);
    }
    if (p_new < pmin) {
      p_new = pmin;
    }

    Real_t q_tilde;
    if (delvc > Real_t(0.0)) {
      q_tilde = Real_t(0.0);
    } else {
      Real_t ssc = (pbvc * e_new + vnewc * vnewc * bvc * p_new) / rho0;
      if (ssc <= Real_t(.1111111e-36)) {
        ssc = Real_t(.3333333e-18);
      } else {
        ssc = sqrt(ssc);
      }
      q_tilde = (ssc * ql_old + qq_old);
    }

    const Real_t sixth = Real_t(1.0) / Real_t(6.0);
    e_new = e_new -
            (Real_t(7.0) * (p_old + q_old) -
             Real_t(8.0) * (pHalfStep + q_new) + (p_new + q_tilde)) *
                delvc * sixth;

    if (fabs(e_new) < e_cut) {
      e_new = Real_t(0.0);
    }
    if (e_new < emin) {
      e_new = emin;
    }

    p_new = bvc * e_new;
    if (fabs(p_new) < p_cut) {
      p_new = Real_t(0.0);
    }
    if (vnewc >= eosvmax) {
      p_new = Real_t(0.0);
    }
    if (p_new < pmin) {
      p_new = pmin;
    }

    if (delvc <= Real_t(0.0)) {
      Real_t ssc = (pbvc * e_new + vnewc * vnewc * bvc * p_new) / rho0;
      if (ssc <= Real_t(.1111111e-36)) {
        ssc = Real_t(.3333333e-18);
      } else {
        ssc = sqrt(ssc);
      }

      q_new = (ssc * ql_old + qq_old);
      if (fabs(q_new) < q_cut) {
        q_new = Real_t(0.0);
      }
    }

    e[ielem] = e_new;
    p[ielem] = p_new;
    q[ielem] = q_new;

    Real_t ssTmp =
        (pbvc * e_new + vnewc * vnewc * bvc * p_new) / rho0;
    if (ssTmp <= Real_t(.1111111e-36)) {
      ssTmp = Real_t(.3333333e-18);
    } else {
      ssTmp = sqrt(ssTmp);
    }
    ss[ielem] = ssTmp;
  }
}

__global__ void KernelUpdateVolumes(Index_t numElem, Real_t v_cut,
                                    const Real_t* vnew, Real_t* v)
{
  Index_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < numElem) {
    Real_t tmpV = vnew[i];
    if (fabs(tmpV - Real_t(1.0)) < v_cut) {
      tmpV = Real_t(1.0);
    }
    v[i] = tmpV;
  }
}

__global__ void KernelCalcCourant(Index_t numElem, const Real_t* arealg,
                                  const Real_t* ss, const Real_t* vdov,
                                  Real_t qqc, Real_t* dtcourant)
{
  Index_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < numElem) {
    Real_t dtf = ss[i] * ss[i];
    if (vdov[i] < Real_t(0.0)) {
      Real_t qqc2 = Real_t(64.0) * qqc * qqc;
      dtf = dtf + qqc2 * arealg[i] * arealg[i] * vdov[i] * vdov[i];
    }
    dtf = sqrt(dtf);
    dtf = arealg[i] / dtf;
    if (vdov[i] != Real_t(0.0)) {
      dtcourant[i] = dtf;
    } else {
      dtcourant[i] = Real_t(1.0e+20);
    }
  }
}

__global__ void KernelCalcHydro(Index_t numElem, const Real_t* vdov,
                                Real_t dvovmax, Real_t* dthydro)
{
  Index_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < numElem) {
    if (vdov[i] != Real_t(0.0)) {
      Real_t dtdvov = dvovmax / (fabs(vdov[i]) + Real_t(1.e-20));
      dthydro[i] = dtdvov;
    } else {
      dthydro[i] = Real_t(1.0e+20);
    }
  }
}

static void AllocateDeviceArray(void** ptr, size_t bytes)
{
  CUDA_CHECK(cudaMalloc(ptr, bytes));
}

static void BuildNodeElementLists(Domain& domain,
                                  std::vector<Index_t>& nodeElemStart,
                                  std::vector<Index_t>& nodeElemCornerList)
{
  Index_t numNode = domain.numNode();
  nodeElemStart.resize(numNode + 1);
  nodeElemStart[0] = 0;
  for (Index_t i = 0; i < numNode; ++i) {
    Index_t count = domain.nodeElemCount(i);
    nodeElemStart[i + 1] = nodeElemStart[i] + count;
  }
  nodeElemCornerList.resize(nodeElemStart[numNode]);
  for (Index_t i = 0; i < numNode; ++i) {
    Index_t count = domain.nodeElemCount(i);
    Index_t* corners = domain.nodeElemCornerList(i);
    Index_t start = nodeElemStart[i];
    for (Index_t j = 0; j < count; ++j) {
      nodeElemCornerList[start + j] = corners[j];
    }
  }
}

static void BuildSymmetryLists(Domain& domain,
                               std::vector<Index_t>& symmX,
                               std::vector<Index_t>& symmY,
                               std::vector<Index_t>& symmZ)
{
  Index_t size = domain.sizeX();
  Index_t numNodeBC = (size + 1) * (size + 1);

  symmX.clear();
  symmY.clear();
  symmZ.clear();

  if (!domain.symmXempty()) {
    symmX.resize(numNodeBC);
    for (Index_t i = 0; i < numNodeBC; ++i) {
      symmX[i] = domain.symmX(i);
    }
  }
  if (!domain.symmYempty()) {
    symmY.resize(numNodeBC);
    for (Index_t i = 0; i < numNodeBC; ++i) {
      symmY[i] = domain.symmY(i);
    }
  }
  if (!domain.symmZempty()) {
    symmZ.resize(numNodeBC);
    for (Index_t i = 0; i < numNodeBC; ++i) {
      symmZ[i] = domain.symmZ(i);
    }
  }
}

static void InitGpuContext(Domain& domain, GpuContext& ctx)
{
  DeviceDomain& d = ctx.d;
  d.numElem = domain.numElem();
  d.numNode = domain.numNode();

  BuildNodeElementLists(domain, ctx.nodeElemStart, ctx.nodeElemCornerList);
  BuildSymmetryLists(domain, ctx.symmX, ctx.symmY, ctx.symmZ);

  d.numSymmX = static_cast<Index_t>(ctx.symmX.size());
  d.numSymmY = static_cast<Index_t>(ctx.symmY.size());
  d.numSymmZ = static_cast<Index_t>(ctx.symmZ.size());

  AllocateDeviceArray(reinterpret_cast<void**>(&d.x), sizeof(Real_t) * d.numNode);
  AllocateDeviceArray(reinterpret_cast<void**>(&d.y), sizeof(Real_t) * d.numNode);
  AllocateDeviceArray(reinterpret_cast<void**>(&d.z), sizeof(Real_t) * d.numNode);
  AllocateDeviceArray(reinterpret_cast<void**>(&d.xd), sizeof(Real_t) * d.numNode);
  AllocateDeviceArray(reinterpret_cast<void**>(&d.yd), sizeof(Real_t) * d.numNode);
  AllocateDeviceArray(reinterpret_cast<void**>(&d.zd), sizeof(Real_t) * d.numNode);
  AllocateDeviceArray(reinterpret_cast<void**>(&d.xdd), sizeof(Real_t) * d.numNode);
  AllocateDeviceArray(reinterpret_cast<void**>(&d.ydd), sizeof(Real_t) * d.numNode);
  AllocateDeviceArray(reinterpret_cast<void**>(&d.zdd), sizeof(Real_t) * d.numNode);
  AllocateDeviceArray(reinterpret_cast<void**>(&d.fx), sizeof(Real_t) * d.numNode);
  AllocateDeviceArray(reinterpret_cast<void**>(&d.fy), sizeof(Real_t) * d.numNode);
  AllocateDeviceArray(reinterpret_cast<void**>(&d.fz), sizeof(Real_t) * d.numNode);
  AllocateDeviceArray(reinterpret_cast<void**>(&d.nodalMass),
                      sizeof(Real_t) * d.numNode);

  AllocateDeviceArray(reinterpret_cast<void**>(&d.e), sizeof(Real_t) * d.numElem);
  AllocateDeviceArray(reinterpret_cast<void**>(&d.p), sizeof(Real_t) * d.numElem);
  AllocateDeviceArray(reinterpret_cast<void**>(&d.q), sizeof(Real_t) * d.numElem);
  AllocateDeviceArray(reinterpret_cast<void**>(&d.ql), sizeof(Real_t) * d.numElem);
  AllocateDeviceArray(reinterpret_cast<void**>(&d.qq), sizeof(Real_t) * d.numElem);
  AllocateDeviceArray(reinterpret_cast<void**>(&d.v), sizeof(Real_t) * d.numElem);
  AllocateDeviceArray(reinterpret_cast<void**>(&d.vnew),
                      sizeof(Real_t) * d.numElem);
  AllocateDeviceArray(reinterpret_cast<void**>(&d.delv),
                      sizeof(Real_t) * d.numElem);
  AllocateDeviceArray(reinterpret_cast<void**>(&d.vdov),
                      sizeof(Real_t) * d.numElem);
  AllocateDeviceArray(reinterpret_cast<void**>(&d.arealg),
                      sizeof(Real_t) * d.numElem);
  AllocateDeviceArray(reinterpret_cast<void**>(&d.volo),
                      sizeof(Real_t) * d.numElem);
  AllocateDeviceArray(reinterpret_cast<void**>(&d.ss), sizeof(Real_t) * d.numElem);
  AllocateDeviceArray(reinterpret_cast<void**>(&d.elemMass),
                      sizeof(Real_t) * d.numElem);

  AllocateDeviceArray(reinterpret_cast<void**>(&d.nodelist),
                      sizeof(Index_t) * d.numElem * 8);
  AllocateDeviceArray(reinterpret_cast<void**>(&d.lxim), sizeof(Index_t) * d.numElem);
  AllocateDeviceArray(reinterpret_cast<void**>(&d.lxip), sizeof(Index_t) * d.numElem);
  AllocateDeviceArray(reinterpret_cast<void**>(&d.letam), sizeof(Index_t) * d.numElem);
  AllocateDeviceArray(reinterpret_cast<void**>(&d.letap), sizeof(Index_t) * d.numElem);
  AllocateDeviceArray(reinterpret_cast<void**>(&d.lzetam), sizeof(Index_t) * d.numElem);
  AllocateDeviceArray(reinterpret_cast<void**>(&d.lzetap), sizeof(Index_t) * d.numElem);
  AllocateDeviceArray(reinterpret_cast<void**>(&d.elemBC), sizeof(Int_t) * d.numElem);

  if (d.numSymmX > 0) {
    AllocateDeviceArray(reinterpret_cast<void**>(&d.symmX),
                        sizeof(Index_t) * d.numSymmX);
  } else {
    d.symmX = NULL;
  }
  if (d.numSymmY > 0) {
    AllocateDeviceArray(reinterpret_cast<void**>(&d.symmY),
                        sizeof(Index_t) * d.numSymmY);
  } else {
    d.symmY = NULL;
  }
  if (d.numSymmZ > 0) {
    AllocateDeviceArray(reinterpret_cast<void**>(&d.symmZ),
                        sizeof(Index_t) * d.numSymmZ);
  } else {
    d.symmZ = NULL;
  }

  AllocateDeviceArray(reinterpret_cast<void**>(&d.nodeElemStart),
                      sizeof(Index_t) * (d.numNode + 1));
  AllocateDeviceArray(reinterpret_cast<void**>(&d.nodeElemCornerList),
                      sizeof(Index_t) * ctx.nodeElemCornerList.size());

  AllocateDeviceArray(reinterpret_cast<void**>(&d.dxx), sizeof(Real_t) * d.numElem);
  AllocateDeviceArray(reinterpret_cast<void**>(&d.dyy), sizeof(Real_t) * d.numElem);
  AllocateDeviceArray(reinterpret_cast<void**>(&d.dzz), sizeof(Real_t) * d.numElem);
  AllocateDeviceArray(reinterpret_cast<void**>(&d.delv_xi), sizeof(Real_t) * d.numElem);
  AllocateDeviceArray(reinterpret_cast<void**>(&d.delv_eta), sizeof(Real_t) * d.numElem);
  AllocateDeviceArray(reinterpret_cast<void**>(&d.delv_zeta), sizeof(Real_t) * d.numElem);
  AllocateDeviceArray(reinterpret_cast<void**>(&d.delx_xi), sizeof(Real_t) * d.numElem);
  AllocateDeviceArray(reinterpret_cast<void**>(&d.delx_eta), sizeof(Real_t) * d.numElem);
  AllocateDeviceArray(reinterpret_cast<void**>(&d.delx_zeta), sizeof(Real_t) * d.numElem);

  AllocateDeviceArray(reinterpret_cast<void**>(&d.determ), sizeof(Real_t) * d.numElem);
  AllocateDeviceArray(reinterpret_cast<void**>(&d.dvdx), sizeof(Real_t) * d.numElem * 8);
  AllocateDeviceArray(reinterpret_cast<void**>(&d.dvdy), sizeof(Real_t) * d.numElem * 8);
  AllocateDeviceArray(reinterpret_cast<void**>(&d.dvdz), sizeof(Real_t) * d.numElem * 8);
  AllocateDeviceArray(reinterpret_cast<void**>(&d.x8n), sizeof(Real_t) * d.numElem * 8);
  AllocateDeviceArray(reinterpret_cast<void**>(&d.y8n), sizeof(Real_t) * d.numElem * 8);
  AllocateDeviceArray(reinterpret_cast<void**>(&d.z8n), sizeof(Real_t) * d.numElem * 8);
  AllocateDeviceArray(reinterpret_cast<void**>(&d.fx_elem), sizeof(Real_t) * d.numElem * 8);
  AllocateDeviceArray(reinterpret_cast<void**>(&d.fy_elem), sizeof(Real_t) * d.numElem * 8);
  AllocateDeviceArray(reinterpret_cast<void**>(&d.fz_elem), sizeof(Real_t) * d.numElem * 8);

  AllocateDeviceArray(reinterpret_cast<void**>(&d.dtcourant_per_elem),
                      sizeof(Real_t) * d.numElem);
  AllocateDeviceArray(reinterpret_cast<void**>(&d.dthydro_per_elem),
                      sizeof(Real_t) * d.numElem);

  AllocateDeviceArray(reinterpret_cast<void**>(&d.error), sizeof(int));
}

static void CopyDomainToDevice(Domain& domain, GpuContext& ctx)
{
  DeviceDomain& d = ctx.d;
  CUDA_CHECK(cudaMemcpy(d.x, &domain.x(0), sizeof(Real_t) * d.numNode,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d.y, &domain.y(0), sizeof(Real_t) * d.numNode,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d.z, &domain.z(0), sizeof(Real_t) * d.numNode,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d.xd, &domain.xd(0), sizeof(Real_t) * d.numNode,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d.yd, &domain.yd(0), sizeof(Real_t) * d.numNode,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d.zd, &domain.zd(0), sizeof(Real_t) * d.numNode,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d.xdd, &domain.xdd(0), sizeof(Real_t) * d.numNode,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d.ydd, &domain.ydd(0), sizeof(Real_t) * d.numNode,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d.zdd, &domain.zdd(0), sizeof(Real_t) * d.numNode,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d.fx, &domain.fx(0), sizeof(Real_t) * d.numNode,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d.fy, &domain.fy(0), sizeof(Real_t) * d.numNode,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d.fz, &domain.fz(0), sizeof(Real_t) * d.numNode,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d.nodalMass, &domain.nodalMass(0),
                        sizeof(Real_t) * d.numNode, cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMemcpy(d.e, &domain.e(0), sizeof(Real_t) * d.numElem,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d.p, &domain.p(0), sizeof(Real_t) * d.numElem,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d.q, &domain.q(0), sizeof(Real_t) * d.numElem,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d.ql, &domain.ql(0), sizeof(Real_t) * d.numElem,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d.qq, &domain.qq(0), sizeof(Real_t) * d.numElem,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d.v, &domain.v(0), sizeof(Real_t) * d.numElem,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d.vnew, &domain.vnew(0), sizeof(Real_t) * d.numElem,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d.delv, &domain.delv(0), sizeof(Real_t) * d.numElem,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d.vdov, &domain.vdov(0), sizeof(Real_t) * d.numElem,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d.arealg, &domain.arealg(0), sizeof(Real_t) * d.numElem,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d.volo, &domain.volo(0), sizeof(Real_t) * d.numElem,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d.ss, &domain.ss(0), sizeof(Real_t) * d.numElem,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d.elemMass, &domain.elemMass(0),
                        sizeof(Real_t) * d.numElem, cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMemcpy(d.nodelist, domain.nodelist(0),
                        sizeof(Index_t) * d.numElem * 8,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d.lxim, &domain.lxim(0), sizeof(Index_t) * d.numElem,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d.lxip, &domain.lxip(0), sizeof(Index_t) * d.numElem,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d.letam, &domain.letam(0), sizeof(Index_t) * d.numElem,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d.letap, &domain.letap(0), sizeof(Index_t) * d.numElem,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d.lzetam, &domain.lzetam(0), sizeof(Index_t) * d.numElem,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d.lzetap, &domain.lzetap(0), sizeof(Index_t) * d.numElem,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d.elemBC, &domain.elemBC(0), sizeof(Int_t) * d.numElem,
                        cudaMemcpyHostToDevice));

  if (d.numSymmX > 0) {
    CUDA_CHECK(cudaMemcpy(d.symmX, ctx.symmX.data(),
                          sizeof(Index_t) * d.numSymmX,
                          cudaMemcpyHostToDevice));
  }
  if (d.numSymmY > 0) {
    CUDA_CHECK(cudaMemcpy(d.symmY, ctx.symmY.data(),
                          sizeof(Index_t) * d.numSymmY,
                          cudaMemcpyHostToDevice));
  }
  if (d.numSymmZ > 0) {
    CUDA_CHECK(cudaMemcpy(d.symmZ, ctx.symmZ.data(),
                          sizeof(Index_t) * d.numSymmZ,
                          cudaMemcpyHostToDevice));
  }

  CUDA_CHECK(cudaMemcpy(d.nodeElemStart, ctx.nodeElemStart.data(),
                        sizeof(Index_t) * (d.numNode + 1),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d.nodeElemCornerList, ctx.nodeElemCornerList.data(),
                        sizeof(Index_t) * ctx.nodeElemCornerList.size(),
                        cudaMemcpyHostToDevice));
}

static void CopyDeviceToHostNodal(const DeviceDomain& d, Domain& domain)
{
  CUDA_CHECK(cudaMemcpy(&domain.x(0), d.x, sizeof(Real_t) * d.numNode,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&domain.y(0), d.y, sizeof(Real_t) * d.numNode,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&domain.z(0), d.z, sizeof(Real_t) * d.numNode,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&domain.xd(0), d.xd, sizeof(Real_t) * d.numNode,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&domain.yd(0), d.yd, sizeof(Real_t) * d.numNode,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&domain.zd(0), d.zd, sizeof(Real_t) * d.numNode,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&domain.xdd(0), d.xdd, sizeof(Real_t) * d.numNode,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&domain.ydd(0), d.ydd, sizeof(Real_t) * d.numNode,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&domain.zdd(0), d.zdd, sizeof(Real_t) * d.numNode,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&domain.fx(0), d.fx, sizeof(Real_t) * d.numNode,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&domain.fy(0), d.fy, sizeof(Real_t) * d.numNode,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&domain.fz(0), d.fz, sizeof(Real_t) * d.numNode,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&domain.nodalMass(0), d.nodalMass,
                        sizeof(Real_t) * d.numNode, cudaMemcpyDeviceToHost));
}

static void CopyDeviceToHostElement(const DeviceDomain& d, Domain& domain)
{
  CUDA_CHECK(cudaMemcpy(&domain.e(0), d.e, sizeof(Real_t) * d.numElem,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&domain.p(0), d.p, sizeof(Real_t) * d.numElem,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&domain.q(0), d.q, sizeof(Real_t) * d.numElem,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&domain.ql(0), d.ql, sizeof(Real_t) * d.numElem,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&domain.qq(0), d.qq, sizeof(Real_t) * d.numElem,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&domain.v(0), d.v, sizeof(Real_t) * d.numElem,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&domain.vnew(0), d.vnew, sizeof(Real_t) * d.numElem,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&domain.volo(0), d.volo, sizeof(Real_t) * d.numElem,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&domain.delv(0), d.delv, sizeof(Real_t) * d.numElem,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&domain.vdov(0), d.vdov, sizeof(Real_t) * d.numElem,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&domain.arealg(0), d.arealg, sizeof(Real_t) * d.numElem,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&domain.ss(0), d.ss, sizeof(Real_t) * d.numElem,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&domain.elemMass(0), d.elemMass,
                        sizeof(Real_t) * d.numElem, cudaMemcpyDeviceToHost));
}

static void FreeGpuContext(GpuContext& ctx)
{
  DeviceDomain& d = ctx.d;
  cudaFree(d.x);
  cudaFree(d.y);
  cudaFree(d.z);
  cudaFree(d.xd);
  cudaFree(d.yd);
  cudaFree(d.zd);
  cudaFree(d.xdd);
  cudaFree(d.ydd);
  cudaFree(d.zdd);
  cudaFree(d.fx);
  cudaFree(d.fy);
  cudaFree(d.fz);
  cudaFree(d.nodalMass);

  cudaFree(d.e);
  cudaFree(d.p);
  cudaFree(d.q);
  cudaFree(d.ql);
  cudaFree(d.qq);
  cudaFree(d.v);
  cudaFree(d.vnew);
  cudaFree(d.delv);
  cudaFree(d.vdov);
  cudaFree(d.arealg);
  cudaFree(d.volo);
  cudaFree(d.ss);
  cudaFree(d.elemMass);

  cudaFree(d.nodelist);
  cudaFree(d.lxim);
  cudaFree(d.lxip);
  cudaFree(d.letam);
  cudaFree(d.letap);
  cudaFree(d.lzetam);
  cudaFree(d.lzetap);
  cudaFree(d.elemBC);

  cudaFree(d.symmX);
  cudaFree(d.symmY);
  cudaFree(d.symmZ);

  cudaFree(d.nodeElemStart);
  cudaFree(d.nodeElemCornerList);

  cudaFree(d.dxx);
  cudaFree(d.dyy);
  cudaFree(d.dzz);
  cudaFree(d.delv_xi);
  cudaFree(d.delv_eta);
  cudaFree(d.delv_zeta);
  cudaFree(d.delx_xi);
  cudaFree(d.delx_eta);
  cudaFree(d.delx_zeta);

  cudaFree(d.determ);
  cudaFree(d.dvdx);
  cudaFree(d.dvdy);
  cudaFree(d.dvdz);
  cudaFree(d.x8n);
  cudaFree(d.y8n);
  cudaFree(d.z8n);
  cudaFree(d.fx_elem);
  cudaFree(d.fy_elem);
  cudaFree(d.fz_elem);

  cudaFree(d.dtcourant_per_elem);
  cudaFree(d.dthydro_per_elem);

  cudaFree(d.error);
}

static void CheckDeviceError(const DeviceDomain& d)
{
  int host_error = 0;
  CUDA_CHECK(cudaMemcpy(&host_error, d.error, sizeof(int),
                        cudaMemcpyDeviceToHost));
  if (host_error != 0) {
    std::fprintf(stderr, "LULESH GPU error: %d\n", host_error);
    std::exit(host_error);
  }
}

static void LagrangeLeapFrogGpu(Domain& domain, GpuContext& ctx)
{
  const LogConfig& log_cfg = GetLogConfig();
  const bool do_log = ShouldLogStep(log_cfg, domain);
  const Index_t log_cycle = domain.cycle();

  DeviceDomain& d = ctx.d;
  const int block_size = 128;

  CUDA_CHECK(cudaMemset(d.error, 0, sizeof(int)));

  if (do_log && log_cfg.log_pre) {
    CopyDeviceToHostNodal(d, domain);
    LogNodalFields(log_cfg, domain,
                   StepNameWithCycle("step0_pre_lagrange_nodal", log_cycle));
  }

  int node_blocks = static_cast<int>((d.numNode + block_size - 1) / block_size);
  int elem_blocks = static_cast<int>((d.numElem + block_size - 1) / block_size);

  KernelZeroForces<<<node_blocks, block_size>>>(d.numNode, d.fx, d.fy, d.fz);
  CUDA_CHECK(cudaGetLastError());

  KernelIntegrateStress<<<elem_blocks, block_size>>>(
      d.numElem, d.nodelist, d.x, d.y, d.z, d.p, d.q, d.determ, d.fx_elem,
      d.fy_elem, d.fz_elem, d.error);
  CUDA_CHECK(cudaGetLastError());

  KernelGatherForces<<<node_blocks, block_size>>>(
      d.numNode, d.nodeElemStart, d.nodeElemCornerList, d.fx_elem, d.fy_elem,
      d.fz_elem, d.fx, d.fy, d.fz, 0);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaDeviceSynchronize());
  CheckDeviceError(d);

  if (do_log && log_cfg.log_substeps) {
    CopyDeviceToHostNodal(d, domain);
    LogNodalFields(log_cfg, domain,
                   StepNameWithCycle("step1a1_post_integrate_stress", log_cycle));
  }

  KernelCalcHourglassInputs<<<elem_blocks, block_size>>>(
      d.numElem, d.nodelist, d.x, d.y, d.z, d.v, d.volo, d.determ, d.dvdx,
      d.dvdy, d.dvdz, d.x8n, d.y8n, d.z8n, d.error);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaDeviceSynchronize());
  CheckDeviceError(d);

  if (do_log && log_cfg.log_substeps) {
    std::vector<Real_t> dvdx(d.numElem * 8);
    std::vector<Real_t> dvdy(d.numElem * 8);
    std::vector<Real_t> dvdz(d.numElem * 8);
    std::vector<Real_t> x8n(d.numElem * 8);
    std::vector<Real_t> y8n(d.numElem * 8);
    std::vector<Real_t> z8n(d.numElem * 8);

    CUDA_CHECK(cudaMemcpy(dvdx.data(), d.dvdx, sizeof(Real_t) * d.numElem * 8,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dvdy.data(), d.dvdy, sizeof(Real_t) * d.numElem * 8,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dvdz.data(), d.dvdz, sizeof(Real_t) * d.numElem * 8,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(x8n.data(), d.x8n, sizeof(Real_t) * d.numElem * 8,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(y8n.data(), d.y8n, sizeof(Real_t) * d.numElem * 8,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(z8n.data(), d.z8n, sizeof(Real_t) * d.numElem * 8,
                          cudaMemcpyDeviceToHost));

    LogHourglassArrays(log_cfg, domain,
                       StepNameWithCycle("step1a1a_hourglass_inputs",
                                         log_cycle),
                       dvdx.data(), dvdy.data(), dvdz.data(), x8n.data(),
                       y8n.data(), z8n.data(), d.numElem * 8);
  }

  Real_t hgcoef = domain.hgcoef();
  if (hgcoef > Real_t(0.0)) {
    KernelCalcHourglassForces<<<elem_blocks, block_size>>>(
        d.numElem, d.nodelist, d.xd, d.yd, d.zd, d.ss, d.elemMass, d.determ,
        d.dvdx, d.dvdy, d.dvdz, d.x8n, d.y8n, d.z8n, hgcoef, d.fx_elem,
        d.fy_elem, d.fz_elem);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());

    if (do_log && log_cfg.log_substeps) {
      std::vector<Real_t> hgfx(d.numElem * 8);
      std::vector<Real_t> hgfy(d.numElem * 8);
      std::vector<Real_t> hgfz(d.numElem * 8);

      CUDA_CHECK(cudaMemcpy(hgfx.data(), d.fx_elem,
                            sizeof(Real_t) * d.numElem * 8,
                            cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(hgfy.data(), d.fy_elem,
                            sizeof(Real_t) * d.numElem * 8,
                            cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(hgfz.data(), d.fz_elem,
                            sizeof(Real_t) * d.numElem * 8,
                            cudaMemcpyDeviceToHost));

      LogHourglassForces(log_cfg, domain,
                         StepNameWithCycle("step1a1b_hourglass_forces",
                                           log_cycle),
                         hgfx.data(), hgfy.data(), hgfz.data(),
                         d.numElem * 8);
    }

    KernelGatherForces<<<node_blocks, block_size>>>(
        d.numNode, d.nodeElemStart, d.nodeElemCornerList, d.fx_elem, d.fy_elem,
        d.fz_elem, d.fx, d.fy, d.fz, 1);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  if (do_log && log_cfg.log_substeps) {
    CopyDeviceToHostNodal(d, domain);
    LogNodalFields(log_cfg, domain,
                   StepNameWithCycle("step1a_post_force", log_cycle));
  }

  KernelCalcAcceleration<<<node_blocks, block_size>>>(
      d.numNode, d.fx, d.fy, d.fz, d.nodalMass, d.xdd, d.ydd, d.zdd);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaDeviceSynchronize());

  if (do_log && log_cfg.log_substeps) {
    CopyDeviceToHostNodal(d, domain);
    LogNodalFields(log_cfg, domain,
                   StepNameWithCycle("step1b_post_accel", log_cycle));
  }

  if (d.numSymmX > 0) {
    int bc_blocks = static_cast<int>((d.numSymmX + block_size - 1) / block_size);
    KernelApplyAccelBC<<<bc_blocks, block_size>>>(d.numSymmX, d.symmX, d.xdd);
  }
  if (d.numSymmY > 0) {
    int bc_blocks = static_cast<int>((d.numSymmY + block_size - 1) / block_size);
    KernelApplyAccelBC<<<bc_blocks, block_size>>>(d.numSymmY, d.symmY, d.ydd);
  }
  if (d.numSymmZ > 0) {
    int bc_blocks = static_cast<int>((d.numSymmZ + block_size - 1) / block_size);
    KernelApplyAccelBC<<<bc_blocks, block_size>>>(d.numSymmZ, d.symmZ, d.zdd);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  if (do_log && log_cfg.log_substeps) {
    CopyDeviceToHostNodal(d, domain);
    LogNodalFields(log_cfg, domain,
                   StepNameWithCycle("step1c_post_accel_bc", log_cycle));
  }

  KernelCalcVelocity<<<node_blocks, block_size>>>(
      d.numNode, domain.deltatime(), domain.u_cut(), d.xd, d.yd, d.zd, d.xdd,
      d.ydd, d.zdd);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  if (do_log && log_cfg.log_substeps) {
    CopyDeviceToHostNodal(d, domain);
    LogNodalFields(log_cfg, domain,
                   StepNameWithCycle("step1d_post_velocity", log_cycle));
  }

  KernelCalcPosition<<<node_blocks, block_size>>>(d.numNode, domain.deltatime(),
                                                  d.x, d.y, d.z, d.xd, d.yd,
                                                  d.zd);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  if (do_log && log_cfg.log_substeps) {
    CopyDeviceToHostNodal(d, domain);
    LogNodalFields(log_cfg, domain,
                   StepNameWithCycle("step1e_post_position", log_cycle));
  }

  if (do_log) {
    CopyDeviceToHostNodal(d, domain);
    LogNodalFields(log_cfg, domain,
                   StepNameWithCycle("step1_post_lagrange_nodal", log_cycle));
  }

  KernelCalcKinematics<<<elem_blocks, block_size>>>(
      d.numElem, d.nodelist, d.x, d.y, d.z, d.xd, d.yd, d.zd, d.volo, d.v,
      domain.deltatime(), d.vnew, d.delv, d.arealg, d.dxx, d.dyy, d.dzz);
  CUDA_CHECK(cudaGetLastError());

  KernelCalcLagrangeElements<<<elem_blocks, block_size>>>(d.numElem, d.dxx,
                                                          d.dyy, d.dzz, d.vnew,
                                                          d.vdov, d.error);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CheckDeviceError(d);

  KernelCalcMonotonicQGradients<<<elem_blocks, block_size>>>(
      d.numElem, d.nodelist, d.x, d.y, d.z, d.xd, d.yd, d.zd, d.volo, d.vnew,
      d.delv_xi, d.delv_eta, d.delv_zeta, d.delx_xi, d.delx_eta, d.delx_zeta);
  CUDA_CHECK(cudaGetLastError());

  KernelCalcMonotonicQ<<<elem_blocks, block_size>>>(
      d.numElem, d.elemBC, d.lxim, d.lxip, d.letam, d.letap, d.lzetam,
      d.lzetap, d.delv_xi, d.delv_eta, d.delv_zeta, d.delx_xi, d.delx_eta,
      d.delx_zeta, d.vdov, d.elemMass, d.volo, d.vnew, domain.monoq_limiter_mult(),
      domain.monoq_max_slope(), domain.qlc_monoq(), domain.qqc_monoq(), d.qq,
      d.ql);
  CUDA_CHECK(cudaGetLastError());

  KernelCheckQStop<<<elem_blocks, block_size>>>(d.numElem, d.q, domain.qstop(),
                                                d.error);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CheckDeviceError(d);

  KernelEvalEOS<<<elem_blocks, block_size>>>(
      d.numElem, d.vnew, d.delv, d.ql, d.qq, d.v, d.volo, d.elemMass,
      domain.e_cut(), domain.p_cut(), domain.q_cut(), domain.pmin(),
      domain.emin(), domain.eosvmax(), domain.eosvmin(), domain.refdens(),
      domain.ss4o3(), d.e, d.p, d.q, d.ss, d.error);
  CUDA_CHECK(cudaGetLastError());

  KernelUpdateVolumes<<<elem_blocks, block_size>>>(d.numElem, domain.v_cut(),
                                                   d.vnew, d.v);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaDeviceSynchronize());
  CheckDeviceError(d);

  if (do_log) {
    CopyDeviceToHostElement(d, domain);
    LogElementFields(log_cfg, domain,
                     StepNameWithCycle("step2_post_lagrange_elements", log_cycle));
  }

  KernelCalcCourant<<<elem_blocks, block_size>>>(d.numElem, d.arealg, d.ss,
                                                 d.vdov, domain.qqc(),
                                                 d.dtcourant_per_elem);
  CUDA_CHECK(cudaGetLastError());

  KernelCalcHydro<<<elem_blocks, block_size>>>(d.numElem, d.vdov,
                                               domain.dvovmax(),
                                               d.dthydro_per_elem);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  thrust::device_ptr<Real_t> dtc_ptr =
      thrust::device_pointer_cast(d.dtcourant_per_elem);
  thrust::device_ptr<Real_t> dth_ptr =
      thrust::device_pointer_cast(d.dthydro_per_elem);

  Real_t dtcourant = thrust::reduce(
      dtc_ptr, dtc_ptr + d.numElem, Real_t(1.0e+20),
      thrust::minimum<Real_t>());
  Real_t dthydro = thrust::reduce(
      dth_ptr, dth_ptr + d.numElem, Real_t(1.0e+20),
      thrust::minimum<Real_t>());

  domain.dtcourant() = dtcourant;
  domain.dthydro() = dthydro;

  if (do_log) {
    LogTimeConstraintFields(log_cfg, domain,
                            StepNameWithCycle("step3_post_time_constraints",
                                              log_cycle));
  }
}

int main(int argc, char* argv[])
{
  Domain* locDom;
  int numRanks = 1;
  int myRank = 0;
  cmdLineOpts opts;

  g_myRank = myRank;
  g_numRanks = numRanks;

  opts.its = 9999999;
  opts.nx = 30;
  opts.numReg = 11;
  opts.numFiles = (int)(numRanks + 10) / 9;
  opts.showProg = 0;
  opts.quiet = 0;
  opts.viz = 0;
  opts.balance = 1;
  opts.cost = 1;

  ParseCommandLineOptions(argc, argv, myRank, &opts);
  g_cmdline_opts = opts;
  g_cmdline_opts_set = true;

  if ((myRank == 0) && (opts.quiet == 0)) {
    std::cout << "Running problem size " << opts.nx
              << "^3 per domain until completion\n";
    std::cout << "Num processors: " << numRanks << "\n";
    std::cout << "Total number of elements: "
              << ((Int8_t)numRanks * opts.nx * opts.nx * opts.nx) << " \n\n";
    std::cout << "To run other sizes, use -s <integer>.\n";
    std::cout << "To run a fixed number of iterations, use -i <integer>.\n";
    std::cout << "To run a more or less balanced region set, use -b <integer>.\n";
    std::cout << "To change the relative costs of regions, use -c <integer>.\n";
    std::cout << "To print out progress, use -p\n";
    std::cout << "To write an output file for VisIt, use -v\n";
    std::cout << "See help (-h) for more options\n\n";
  }

  Int_t col = 0;
  Int_t row = 0;
  Int_t plane = 0;
  Int_t side = 0;
  InitMeshDecomp(numRanks, myRank, &col, &row, &plane, &side);

  locDom = new Domain(numRanks, col, row, plane, opts.nx, side, opts.numReg,
                      opts.balance, opts.cost);

  GpuContext gpu;
  InitGpuContext(*locDom, gpu);
  CopyDomainToDevice(*locDom, gpu);

  timeval start;
  gettimeofday(&start, NULL);

  while ((locDom->time() < locDom->stoptime()) &&
         (locDom->cycle() < opts.its)) {
    TimeIncrement(*locDom);
    LagrangeLeapFrogGpu(*locDom, gpu);

    if ((opts.showProg != 0) && (opts.quiet == 0) && (myRank == 0)) {
      std::cout << "cycle = " << locDom->cycle() << ", "
                << std::scientific << "time = " << double(locDom->time())
                << ", " << "dt=" << double(locDom->deltatime()) << "\n";
      std::cout.unsetf(std::ios_base::floatfield);
    }
  }

  timeval end;
  gettimeofday(&end, NULL);
  double elapsed_time = (double)(end.tv_sec - start.tv_sec) +
                        ((double)(end.tv_usec - start.tv_usec)) / 1000000;
  double elapsed_timeG = elapsed_time;

  CopyDeviceToHostNodal(gpu.d, *locDom);
  CopyDeviceToHostElement(gpu.d, *locDom);

  if (opts.viz) {
    DumpToVisit(*locDom, opts.numFiles, myRank, numRanks);
  }

  if ((myRank == 0) && (opts.quiet == 0)) {
    VerifyAndWriteFinalOutput(elapsed_timeG, *locDom, opts.nx, numRanks);
  }

  FreeGpuContext(gpu);
  delete locDom;
  return 0;
}
