import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { validateModelSelection, exactModelKey } from "../model-selection.js";
const fixtures = JSON.parse(
  readFileSync(
    new URL("../../../fixtures/model-selection/golden.json", import.meta.url),
    "utf8",
  ),
);
describe("shared selection fixtures", () => {
  for (const fixture of fixtures)
    it(fixture.name, () => {
      const call = () =>
        validateModelSelection(fixture.selection, fixture.preferred);
      if (fixture.valid) expect(call).not.toThrow();
      else expect(call).toThrow();
    });
  it("file set identity is ordered deterministically", () => {
    const ref = {
      provider: "huggingface",
      repository: "a/b",
      revision: "a".repeat(40),
    };
    expect(exactModelKey({ ...ref, files: ["b", "a"] })).toBe(
      exactModelKey({ ...ref, files: ["a", "b"] }),
    );
    expect(exactModelKey({ ...ref, files: ["a"] })).not.toBe(
      exactModelKey({ ...ref, files: ["b"] }),
    );
  });
});
