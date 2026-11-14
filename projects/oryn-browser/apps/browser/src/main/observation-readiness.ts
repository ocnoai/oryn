export const OBSERVATION_READY_TIMEOUT_MS = 2000;

/** Failed observation cannot block browsing, or turn missing evidence into Secure. */
export function boundedObservationReady(ready: void | Promise<void>): Promise<void> | undefined {
  if (!ready) return undefined;
  return new Promise((resolve) => {
    const timer = setTimeout(resolve, OBSERVATION_READY_TIMEOUT_MS);
    const finish = () => {
      clearTimeout(timer);
      resolve();
    };
    void ready.then(finish, finish);
  });
}
