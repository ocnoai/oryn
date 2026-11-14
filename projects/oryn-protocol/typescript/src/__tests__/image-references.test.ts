import { expect, it } from "vitest";
import profiles from "../../../fixtures/task-contracts/image-references.json";
import { validateTaskInvocation, type TaskProfile } from "../task-contracts.js";
const ids = [..."abcd"].map(c => "artifact_" + c.repeat(24));
const base = { imageArtifactId: ids[0], prompt: "Use image 1 and image 2" };
it.each([["legacy", [0]], ["optional", [0, 1, 2]], ["pair", [1]]] as const)("enforces %s reference counts", (name, counts) => {
  const profile = profiles[name] as unknown as TaskProfile;
  for (let count = 0; count < 4; count++) {
    const value = { ...base, ...(count ? { referenceImageArtifactIds: ids.slice(1, count + 1) } : {}) };
    const check = () => validateTaskInvocation(profile, value);
    if ((counts as readonly number[]).includes(count)) expect(check).not.toThrow(); else expect(check).toThrow();
  }
});
it.each([[ids[0]], [ids[1], ids[1]], ["/tmp/image.png"], "bad", null, [3]].map(references => ({references})))("rejects malformed references $references", ({references}) => {
  expect(() => validateTaskInvocation(profiles.optional as unknown as TaskProfile, { ...base, referenceImageArtifactIds: references })).toThrow();
});
