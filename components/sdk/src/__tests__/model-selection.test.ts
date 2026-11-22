import { describe, it, expect } from "vitest";
import fixtures from "./fixtures/model-selection.json";
import {
  validateModelSelection,
  type SelectionModelRef,
} from "../model-selection.js";
describe("Protocol selection contract", () => {
  for (const fixture of fixtures)
    it(fixture.name, () => {
      const call = () =>
        validateModelSelection(
          fixture.selection,
          fixture.preferred as SelectionModelRef | undefined,
        );
      if (fixture.valid) expect(call).not.toThrow();
      else expect(call).toThrow();
    });
});
