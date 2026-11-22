/* Trusted Browser import helper. Loaded before untrusted import data is decoded. */
#define NODE_GYP_MODULE_NAME browser_import_sandbox
#include <node_api.h>
#include <sandbox.h>
#include <sys/resource.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <sys/stat.h>

static napi_value fail(napi_env env) {
  napi_throw_error(env, "IMPORT_SANDBOX_UNAVAILABLE", "Browser import isolation is unavailable.");
  return NULL;
}

static int limit_resource(int resource, rlim_t limit) {
  struct rlimit current;
  if (getrlimit(resource, &current) != 0) return -1;
  struct rlimit bounded = {limit, limit};
  if (current.rlim_max != RLIM_INFINITY && current.rlim_max < limit)
    bounded.rlim_cur = bounded.rlim_max = current.rlim_max;
  return setrlimit(resource, &bounded);
}

static napi_value confine(napi_env env, napi_callback_info info) {
  size_t argc = 4;
  napi_value argv[4];
  size_t length = 0;
  uint32_t cpu_seconds = 10;
  uint32_t file_bytes = 0;
  if (napi_get_cb_info(env, info, &argc, argv, NULL, NULL) != napi_ok || argc < 1 ||
      napi_get_value_string_utf8(env, argv[0], NULL, 0, &length) != napi_ok ||
      length == 0 || length > 65536) return fail(env);
  if (argc > 1 && (napi_get_value_uint32(env, argv[1], &cpu_seconds) != napi_ok || cpu_seconds < 1 || cpu_seconds > 30)) return fail(env);
  if (argc > 2 && (napi_get_value_uint32(env, argv[2], &file_bytes) != napi_ok || file_bytes > 16777216)) return fail(env);
  char *profile = malloc(length + 1);
  if (!profile) return fail(env);
  if (napi_get_value_string_utf8(env, argv[0], profile, length + 1, &length) != napi_ok) { free(profile); return fail(env); }
  napi_valuetype exact_type = napi_undefined;
  if (argc > 3 && napi_typeof(env, argv[3], &exact_type) != napi_ok) { free(profile); return fail(env); }
  if (exact_type != napi_undefined) {
    char selected[PATH_MAX], canonical[PATH_MAX];
    size_t selected_length = 0;
    struct stat information;
    if (exact_type != napi_string || napi_get_value_string_utf8(env, argv[3], NULL, 0, &selected_length) != napi_ok ||
        selected_length == 0 || selected_length >= sizeof(selected) ||
        napi_get_value_string_utf8(env, argv[3], selected, sizeof(selected), &selected_length) != napi_ok ||
        strlen(selected) != selected_length || selected[0] != '/' ||
        !realpath(selected, canonical) || strcmp(selected, canonical) != 0 ||
        lstat(selected, &information) != 0 || !S_ISREG(information.st_mode) ||
        information.st_size < 0 || information.st_size > 67108864) { free(profile); return fail(env); }
    for (size_t index = 0; index < selected_length; index++)
      if ((unsigned char)selected[index] < 32 || (unsigned char)selected[index] == 127) { free(profile); return fail(env); }
    const char *prefix = "\n(allow file-read* (literal \"";
    const char *suffix = "\"))\n";
    char *expanded = realloc(profile, length + selected_length * 2 + strlen(prefix) + strlen(suffix) + 1);
    if (!expanded) { free(profile); return fail(env); }
    profile = expanded;
    memcpy(profile + length, prefix, strlen(prefix)); length += strlen(prefix);
    for (size_t index = 0; index < selected_length; index++) {
      if (selected[index] == '\\' || selected[index] == '"') profile[length++] = '\\';
      profile[length++] = selected[index];
    }
    memcpy(profile + length, suffix, strlen(suffix) + 1);
  }
  /* RSS is supervised by Browser main; do not pretend Darwin's RLIMIT_RSS caps it. */
  if (limit_resource(RLIMIT_CPU, cpu_seconds) != 0 ||
      limit_resource(RLIMIT_FSIZE, file_bytes) != 0 ||
      limit_resource(RLIMIT_CORE, 0) != 0 ||
      limit_resource(RLIMIT_NOFILE, 256) != 0 ||
      limit_resource(RLIMIT_NPROC, 0) != 0) { free(profile); return fail(env); }
  char *error = NULL;
  int result = sandbox_init(profile, 0, &error);
  free(profile);
  if (error) sandbox_free_error(error);
  if (result != 0) return fail(env);
  napi_value success;
  if (napi_get_boolean(env, true, &success) != napi_ok) return fail(env);
  return success;
}

static napi_value initialize(napi_env env, napi_value exports) {
  napi_value method, version;
  if (napi_create_function(env, "confine", NAPI_AUTO_LENGTH, confine, NULL, &method) != napi_ok ||
      napi_set_named_property(env, exports, "confine", method) != napi_ok) return fail(env);
  if (napi_create_uint32(env, 2, &version) != napi_ok || napi_set_named_property(env, exports, "protocolVersion", version) != napi_ok) return fail(env);
  return exports;
}
NAPI_MODULE(NODE_GYP_MODULE_NAME, initialize)
