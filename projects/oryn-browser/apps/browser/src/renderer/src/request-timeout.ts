/** Bound UI waiting even if an IPC reply or engine startup never settles. */
export function withRequestTimeout<T>(request: Promise<T>, milliseconds = 30_000): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('Request timed out')), milliseconds);
    request.then(
      (value) => {
        clearTimeout(timer);
        resolve(value);
      },
      (error: unknown) => {
        clearTimeout(timer);
        reject(error);
      },
    );
  });
}
