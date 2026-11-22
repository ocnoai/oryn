import { readFile } from "node:fs/promises";

import { describe, expect, it } from "vitest";

import {
  assertCandidateExecutionProfile,
  assertEngineSessionAttached,
  assertEngineJobEvent,
  assertEngineJobReference,
  assertModelInstallProgressEvent,
  assertModelInventoryResult,
  assertModelRemovePreviewResult,
  assertModelRemoveResult,
  assertModelResidencyChangedEvent,
  assertHandshakeResponse,
  assertProtocolResponse,
  assertRuntimeCapabilities,
  assertRuntimeMemoryPressureChangedEvent,
  assertRuntimeResourceSnapshot,
  assertRuntimePackCleanupResult,
  assertRuntimePackInventoryResult,
  assertRuntimeResourcesUpdatedEvent,
  assertRuntimeStateEvent,
  ProtocolEventSequence,
  ProtocolLineDecoder,
  ProtocolLineError,
  parseProtocolJSONLine,
} from "../index.js";

async function fixture(name: string): Promise<unknown> {
  const location = new URL(`../../../fixtures/v1/${name}`, import.meta.url);
  return JSON.parse(await readFile(location, "utf8")) as unknown;
}

describe("shared protocol schema", () => {
  it("accepts paired GGUF projector inventory and rejects path escapes", async () => {
    const response = await fixture("model-inventory-response.json");
    assertProtocolResponse(response);
    if (!response.ok) throw new Error("Expected inventory success");
    assertModelInventoryResult(response.result);
    const inventory = response.result;
    inventory.models[0]!.runtimePack = {
      packId: "oryn.llama-cpp", packVersion: "0.2.0", runnerId: "llama.cpp.chat",
      taskContract: "chat-v1", compatibilityLevel: "native-declarative",
      manifestDigest: "a".repeat(64),
      trustedMetadata: {license: "Repository-defined", modelFile: "model.gguf",
        projectorFile: "mmproj-f16.gguf", conformance: {status: "fixture-only", platforms: ["cpu"]}},
    };
    expect(() => assertModelInventoryResult(inventory)).not.toThrow();
    inventory.models[0]!.runtimePack.trustedMetadata!.projectorFile = "../mmproj.gguf";
    expect(() => assertModelInventoryResult(inventory)).toThrow();
  });
  it("accepts the golden handshake responses", async () => {
    const success = await fixture("handshake-success.json");
    const versionError = await fixture("handshake-version-error.json");

    expect(() => assertHandshakeResponse(success)).not.toThrow();
    expect(() => assertHandshakeResponse(versionError)).not.toThrow();
  });

  it("accepts and narrows the runtime capabilities fixture", async () => {
    const response = await fixture("runtime-capabilities.json");
    assertProtocolResponse(response);

    expect(response.ok).toBe(true);
    if (response.ok) {
      assertRuntimeCapabilities(response.result);
      expect(response.result.scheduler.lanes[0]).toEqual({
        id: "generation",
        capacity: 1,
        global: true,
      });
    }
  });

  it("rejects unknown response envelope fields", () => {
    expect(() =>
      assertProtocolResponse({
        v: 1,
        type: "response",
        id: "req_test",
        ok: true,
        result: {},
        untrusted: true,
      }),
    ).toThrow();
  });

  it("validates runtime events and enforces engine-global sequence order", async () => {
    const event = await fixture("runtime-state-event.json");
    assertRuntimeStateEvent(event);
    const sequence = new ProtocolEventSequence();
    sequence.accept(event);
    expect(() => sequence.accept(event)).toThrow("sequence did not increase");
  });

  it("accepts the shutdown and error envelope fixtures", async () => {
    const shutdown = await fixture("runtime-shutdown-response.json");
    const error = await fixture("error-response.json");
    expect(() => assertProtocolResponse(shutdown)).not.toThrow();
    expect(() => assertProtocolResponse(error)).not.toThrow();
  });

  it("validates trusted resource snapshots and pressure transitions", async () => {
    const response = await fixture("runtime-resources.json");
    assertProtocolResponse(response);
    if (!response.ok) throw new Error("Expected resource fixture success.");
    assertRuntimeResourceSnapshot(response.result);
    expect(response.result.memoryPressure).toBe("high");

    const event = await fixture("runtime-memory-pressure-event.json");
    assertRuntimeMemoryPressureChangedEvent(event);
    expect(event.data.current).toBe("high");

    const resourcesEvent = await fixture("runtime-resources-event.json");
    assertRuntimeResourcesUpdatedEvent(resourcesEvent);
    expect(resourcesEvent.data.generationLane.capacity).toBe(1);
  });

  it("accepts bounded optional resource provenance and rejects false metrics", () => {
    const base = {
      snapshotId: "resource_optional",
      timestamp: 1,
      memoryPressure: "normal",
      localAI: { estimatedResidentModelBytes: 100 },
      generationLane: { capacity: 1, running: 1, queued: 2 },
    };
    expect(() =>
      assertRuntimeResourceSnapshot({
        ...base,
        processMemory: {
          residentBytes: 50,
          cpuPercent: 14.5,
          sampleWindowMs: 1000,
        },
        supervisorMemory: { residentBytes: 50 },
        workers: [{ workerId: "worker_one", kind: "mlx", residentBytes: 75 }],
        runningJob: {
          jobId: "job_one",
          model: "org/model",
          task: "chat",
          estimatedBytes: 100,
        },
        accelerator: {
          backend: "mps",
          utilizationPercent: 30,
          sampleWindowMs: 1000,
        },
        throughput: { value: 12.5, unit: "tokens/s", sampleWindowMs: 1000 },
      }),
    ).not.toThrow();
    expect(() =>
      assertRuntimeResourceSnapshot({
        ...base,
        processMemory: { cpuPercent: Number.NaN, sampleWindowMs: -1 },
      }),
    ).toThrow();
    expect(() =>
      assertRuntimeResourceSnapshot({
        ...base,
        throughput: { value: 2, unit: "percent", sampleWindowMs: 1000 },
      }),
    ).toThrow();
  });

  it("validates signed runtime-pack inventory and cleanup results", () => {
    assertRuntimePackInventoryResult({
      catalogVersion: 1,
      catalogDigest: "a".repeat(64),
      thirdPartyPublishersEnabled: false,
      automaticExecutableUpdatesEnabled: false,
      packs: [
        {
          packId: "oryn.llama-cpp",
          version: "0.1.0",
          publisher: "oryn.first-party",
          abi: "1",
          dependencyClosure: "llama.cpp-b10218",
          platforms: ["macos-arm64"],
          distribution: "bundled",
          state: "active",
          bundled: true,
          removable: false,
          sizeBytes: 0,
          minimumSafeVersion: "0.1.0",
          revoked: false,
          license: {
            summary: "MIT native runtime",
            notice: "Upstream notice is packaged with the runtime.",
            access: "public",
            acceptanceRequired: false,
            accepted: true,
          },
        },
      ],
    });
    assertRuntimePackCleanupResult({ removedPacks: 2, reclaimedBytes: 4096 });
    expect(() =>
      assertRuntimePackInventoryResult({
        catalogVersion: 1,
        catalogDigest: "a".repeat(64),
        thirdPartyPublishersEnabled: true,
        automaticExecutableUpdatesEnabled: false,
        packs: [],
      }),
    ).toThrow();
  });

  it("validates model inspection, progress, inventory, and removal fixtures", async () => {
    const inspection = await fixture("model-inspect-response.json");
    assertProtocolResponse(inspection);
    if (!inspection.ok) throw new Error("Expected inspection fixture success.");
    assertCandidateExecutionProfile(inspection.result);
    expect(inspection.result.model.revision).toBe(
      inspection.result.exactRevision,
    );
    expect(inspection.result.selectedFiles).toEqual([
      { name: "config.json", sizeBytes: 10 },
      { name: "model.safetensors", sizeBytes: 90 },
    ]);
    assertCandidateExecutionProfile({
      ...inspection.result,
      generationProfile: {
        profileId: "runtime.generic.fallback",
        provenance: "runtime-fallback",
        recommended: false,
        parameters: {
          width: {
            default: 512,
            minimum: 64,
            maximum: 2048,
            step: 8,
            supported: true,
            editable: true,
          },
          height: {
            default: 512,
            minimum: 64,
            maximum: 2048,
            step: 8,
            supported: true,
            editable: true,
          },
          steps: {
            default: 20,
            minimum: 1,
            maximum: 100,
            step: 1,
            supported: true,
            editable: true,
          },
          guidanceScale: {
            default: 7.5,
            minimum: 0,
            maximum: 30,
            step: 0.1,
            supported: true,
            editable: true,
          },
        },
      },
    });

    assertModelInstallProgressEvent(
      await fixture("model-install-progress.json"),
    );
    const inventory = await fixture("model-inventory-response.json");
    assertProtocolResponse(inventory);
    if (!inventory.ok) throw new Error("Expected inventory fixture success.");
    assertModelInventoryResult(inventory.result);

    const preview = await fixture("model-remove-preview-response.json");
    assertProtocolResponse(preview);
    if (!preview.ok) throw new Error("Expected removal fixture success.");
    assertModelRemovePreviewResult(preview.result);

    const removal = await fixture("model-remove-response.json");
    assertProtocolResponse(removal);
    if (!removal.ok) throw new Error("Expected model remove success.");
    assertModelRemoveResult(removal.result);

    const residency = await fixture("model-residency-event.json");
    assertModelResidencyChangedEvent(residency);
    expect(residency.data.state).toBe("loaded");
  });

  it("bounds private versioned routing facts and requires boolean preparation state", async () => {
    const response = await fixture("model-inspect-response.json");
    assertProtocolResponse(response);
    if (!response.ok) throw new Error("Expected inspection fixture success.");
    const profile = response.result;
    assertCandidateExecutionProfile(profile);
    const offer = {
      engine: "oryn.mlx-lm",
      engineVersion: "0.3.0",
      features: ["streaming"],
    };
    const facts = { version: 1, offers: [offer] };
    expect(() =>
      assertCandidateExecutionProfile({
        ...profile,
        routingFacts: facts,
        runtimePreparationRequired: true,
      }),
    ).not.toThrow();
    for (const invalid of [
      { ...facts, version: 2 },
      { ...facts, offers: [{ ...offer, command: "run" }] },
      {
        ...facts,
        offers: [{ ...offer, features: ["streaming", "streaming"] }],
      },
      { ...facts, offers: Array(257).fill(offer) },
    ]) {
      expect(() =>
        assertCandidateExecutionProfile({
          ...profile,
          routingFacts: invalid,
        }),
      ).toThrow();
    }
    expect(() =>
      assertCandidateExecutionProfile({
        ...profile,
        runtimePreparationRequired: "true",
      }),
    ).toThrow();
  });

  it("validates complete compatibility axes and rejects impossible deferral or disposition", async () => {
    const response = await fixture("model-inspect-response.json");
    assertProtocolResponse(response);
    if (!response.ok) throw new Error("Expected inspection fixture success.");
    assertCandidateExecutionProfile(response.result);
    const eligibility = {
      access: { state: "available", reasons: [] },
      artifacts: { state: "safe", reasons: [] },
      runtime: { state: "supported", reasons: [] },
      task: { state: "supported", reasons: [] },
      resources: { state: "admissible", reasons: [] },
      conformance: { state: "family-conformant", reasons: [] },
    } as const;
    const aware = {
      ...response.result,
      descriptorSchemaVersion: "oryn-model-descriptor/1",
      capabilityRegistryDigest: "b".repeat(64),
      eligibility,
      loadDisposition: "trust-required",
    };
    expect(() => assertCandidateExecutionProfile(aware)).not.toThrow();
    expect(() =>
      assertCandidateExecutionProfile({
        ...aware,
        loadDisposition: "verified-load",
      }),
    ).toThrow("Inconsistent compatibility load disposition");
    expect(() =>
      assertCandidateExecutionProfile({
        ...aware,
        eligibility: {
          ...eligibility,
          conformance: { state: "untested", reasons: ["CONFORMANCE_REQUIRED"] },
        },
        blockingReasons: ["CONFORMANCE_REQUIRED"],
      }),
    ).not.toThrow();
    expect(() =>
      assertCandidateExecutionProfile({
        ...aware,
        eligibility: {
          ...eligibility,
          task: { state: "deferred", reasons: [] },
        },
      }),
    ).toThrow();
  });

  it("validates the engine Model Session attachment fixture", async () => {
    const response = await fixture("session-attached-response.json");
    assertProtocolResponse(response);
    if (!response.ok) throw new Error("Expected session fixture success.");
    assertEngineSessionAttached(response.result);
    expect(response.result.residencyState).toBe("loaded");
  });

  it("validates job queue acknowledgements and ordered terminal events", async () => {
    const response = await fixture("job-queued-response.json");
    assertProtocolResponse(response);
    if (!response.ok) throw new Error("Expected queued job fixture success.");
    assertEngineJobReference(response.result);
    expect(response.result.queue?.position).toBe(1);

    const state = await fixture("job-state-event.json");
    const completed = await fixture("job-completed-event.json");
    const cancelled = await fixture("job-cancelled-event.json");
    assertEngineJobEvent(state);
    assertEngineJobEvent(completed);
    assertEngineJobEvent(cancelled);
    expect(completed.data.sequence).toBeGreaterThan(state.data.sequence);
  });

  it("validates preview, artifact, and transactional multi-result descriptors", () => {
    const artifact = {
      outputId: "output_example",
      internalPath: "/controlled/staging/engine/job/output",
      mimeType: "video/mp4",
      sizeBytes: 64,
      name: "result.mp4",
    };
    assertEngineJobEvent({
      v: 1,
      type: "event",
      event: "job.preview",
      seq: 30,
      timestamp: 100,
      data: {
        jobId: "job_example",
        sessionId: "session_example",
        sequence: 1,
        output: { ...artifact, outputId: "output_preview" },
      },
    });
    assertEngineJobEvent({
      v: 1,
      type: "event",
      event: "job.completed",
      seq: 31,
      timestamp: 101,
      data: {
        jobId: "job_example",
        sessionId: "session_example",
        sequence: 2,
        result: {
          kind: "multi",
          outputs: [
            { kind: "artifact", artifact },
            { kind: "structured", value: { durationSeconds: 1 } },
          ],
        },
      },
    });
  });
});

describe("ProtocolLineDecoder", () => {
  it("frames split and combined JSON Lines", () => {
    const decoder = new ProtocolLineDecoder();

    expect(decoder.push(Buffer.from('{"one":1'))).toEqual([]);
    expect(decoder.push(Buffer.from('}\n{"two":2}\r\n'))).toEqual([
      '{"one":1}',
      '{"two":2}',
    ]);
    expect(() => decoder.finish()).not.toThrow();
  });

  it("rejects an oversized partial line before unbounded buffering", () => {
    const decoder = new ProtocolLineDecoder(8);

    expect(() => decoder.push(Buffer.from("123456789"))).toThrowError(
      ProtocolLineError,
    );
  });

  it("rejects invalid UTF-8 and a truncated final JSON Line", () => {
    const invalid = new ProtocolLineDecoder();
    expect(() => invalid.push(Buffer.from([0xc3, 0x28, 0x0a]))).toThrow(
      "invalid UTF-8",
    );
    const truncated = new ProtocolLineDecoder();
    truncated.push(Buffer.from('{"type":"event"'));
    expect(() => truncated.finish()).toThrow("middle of a protocol line");
  });

  it("rejects invalid JSON, huge nesting, and huge arrays before schema validation", () => {
    expect(() => parseProtocolJSONLine('{"truncated":')).toThrow(
      "invalid JSON",
    );
    expect(() =>
      parseProtocolJSONLine(`${"[".repeat(66)}0${"]".repeat(66)}`),
    ).toThrow("structural complexity");
    expect(() =>
      parseProtocolJSONLine(
        JSON.stringify(Array.from({ length: 50_001 }, () => 0)),
      ),
    ).toThrow("structural complexity");
  });

  it("rejects wrong enums, unknown events, and unexpected result shapes", () => {
    expect(() =>
      assertRuntimeStateEvent({
        v: 1,
        type: "event",
        event: "runtime.state",
        seq: 1,
        timestamp: 1,
        data: { state: "corrupted" },
      }),
    ).toThrow();
    expect(() =>
      assertEngineJobEvent({
        v: 1,
        type: "event",
        event: "job.unknown",
        seq: 2,
        timestamp: 1,
        data: { jobId: "job_x", sessionId: "session_x", sequence: 1 },
      }),
    ).toThrow();
    expect(() =>
      assertRuntimeCapabilities({
        providers: [],
        scheduler: { lanes: "not-an-array" },
      }),
    ).toThrow();
  });
});
