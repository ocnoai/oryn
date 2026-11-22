/* Browser-only download metadata. No URL, command or executable is accepted. */
#define NODE_GYP_MODULE_NAME download_safety
#include <node_api.h>
#import <Foundation/Foundation.h>
#include <sys/stat.h>
#include <sys/xattr.h>
#include <fcntl.h>
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static napi_value fail(napi_env env, const char *code) {
  napi_throw_error(env, code, code);
  return NULL;
}

static int descriptor(napi_env env, napi_callback_info info) {
  size_t count = 1;
  napi_value args[1];
  int32_t fd = -1;
  struct stat value;
  if (napi_get_cb_info(env, info, &count, args, NULL, NULL) != napi_ok || count != 1 ||
      napi_get_value_int32(env, args[0], &fd) != napi_ok || fd < 3 ||
      fstat(fd, &value) != 0 || !S_ISREG(value.st_mode) || value.st_nlink != 1) return -1;
  return fd;
}

/* -1: unavailable/invalid, 0: missing, 1: quarantine bit present. */
static int quarantine(int fd) {
  char value[4097];
  ssize_t size = fgetxattr(fd, "com.apple.quarantine", value, sizeof(value) - 1, 0, 0);
  if (size < 0) return errno == ENOATTR ? 0 : -1;
  if (size < 3 || size > 4096 || memchr(value, '\0', (size_t)size)) return -1;
  value[size] = '\0';
  char *separator = (char *)memchr(value, ';', (size_t)size);
  if (!separator || separator == value || separator - value > 8) return -1;
  for (char *cursor = value; cursor < separator; cursor++)
    if (!((*cursor >= '0' && *cursor <= '9') || (*cursor >= 'a' && *cursor <= 'f') ||
          (*cursor >= 'A' && *cursor <= 'F'))) return -1;
  char *end = NULL;
  unsigned long flags = strtoul(value, &end, 16);
  if (end != separator || !(flags & 1)) return -1;
  return 1;
}

static napi_value ensure(napi_env env, napi_callback_info info) {
  int fd = descriptor(env, info);
  if (fd < 0) return fail(env, "DOWNLOAD_QUARANTINE_UNAVAILABLE");
  int state = quarantine(fd);
  const char *result = "preserved";
  if (state < 0) return fail(env, "DOWNLOAD_QUARANTINE_UNAVAILABLE");
  if (state == 0) {
    char value[96];
    int length = snprintf(value, sizeof(value), "0001;%llx;Oryn Browser;", (unsigned long long)time(NULL));
    if (length < 1 || (size_t)length >= sizeof(value)) return fail(env, "DOWNLOAD_QUARANTINE_UNAVAILABLE");
    /* CREATE never overwrites a native tag arriving concurrently. No approval bits are set. */
    int written = fsetxattr(fd, "com.apple.quarantine", value, (size_t)length, 0, XATTR_CREATE);
    if (written != 0 && errno != EEXIST) return fail(env, "DOWNLOAD_QUARANTINE_UNAVAILABLE");
    if (quarantine(fd) != 1 || fsync(fd) != 0) return fail(env, "DOWNLOAD_QUARANTINE_UNAVAILABLE");
    result = written == 0 ? "added" : "preserved";
  }
  napi_value output;
  napi_create_string_utf8(env, result, NAPI_AUTO_LENGTH, &output);
  return output;
}

static napi_value has(napi_env env, napi_callback_info info) {
  int fd = descriptor(env, info);
  napi_value output;
  napi_get_boolean(env, fd >= 0 && quarantine(fd) == 1, &output);
  return output;
}

/* LaunchServices receives an inode-bound file-reference URL, never a re-opened mutable name. */
static napi_value reference(napi_env env, napi_callback_info info) {
  int fd = descriptor(env, info);
  char path[PATH_MAX];
  struct stat held, target;
  if (fd < 0 || fcntl(fd, F_GETPATH, path) != 0 || fstat(fd, &held) != 0)
    return fail(env, "DOWNLOAD_FILE_CHANGED");
  @autoreleasepool {
    NSString *name = [NSString stringWithUTF8String:path];
    NSURL *url = name ? [[NSURL fileURLWithPath:name isDirectory:NO] fileReferenceURL] : nil;
    /* -path resolves a file-reference URL back to its mutable name. Keep its file-ID spelling. */
    const char *spelling = url ? [[url absoluteString] UTF8String] : NULL;
    const char *resolved = url ? [[url path] fileSystemRepresentation] : NULL;
    if (!spelling || strncmp(spelling, "file:///.file/id=", 17) != 0 || strlen(spelling) >= PATH_MAX ||
        !resolved || stat(resolved, &target) != 0 || held.st_dev != target.st_dev || held.st_ino != target.st_ino)
      return fail(env, "DOWNLOAD_FILE_CHANGED");
    napi_value output;
    napi_create_string_utf8(env, spelling, NAPI_AUTO_LENGTH, &output);
    return output;
  }
}

static napi_value initialize(napi_env env, napi_value exports) {
  napi_property_descriptor methods[] = {
    {"ensureQuarantine", NULL, ensure, NULL, NULL, NULL, napi_default, NULL},
    {"hasQuarantine", NULL, has, NULL, NULL, NULL, napi_default, NULL},
    {"fileReferenceURL", NULL, reference, NULL, NULL, NULL, napi_default, NULL},
  };
  napi_define_properties(env, exports, 3, methods);
  return exports;
}
NAPI_MODULE(NODE_GYP_MODULE_NAME, initialize)
