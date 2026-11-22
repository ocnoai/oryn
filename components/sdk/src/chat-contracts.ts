// Generated from canonical Protocol chat schemas; do not hand-edit.
export type ChatV2Input = { "messages": ({ "role": "system" | "user" | "assistant"; "content": string })[]; "imageArtifactId"?: string; "maxTokens"?: number; "temperature"?: number; "topP"?: number; "thinking"?: boolean };
export type ChatV2Value = { "text": string; "reasoning"?: string; "finishReason": "stop" | "length" | "unknown"; "usage": { "inputTokens": number | null; "outputTokens": number | null; "reasoningTokens": number | null; "answerTokens": number | null }; "metadata": { "thinking": { "requested": boolean | null; "applied": boolean; "effective": boolean | null; "reason": "applied" | "already_fixed" | "model_default" | "unsupported_model" | "fixed_mode" | "unknown_support"; "source": "engine" } } };
export type ChatThinkingCapability = { "support": "switchable" | "fixed" | "unsupported" | "unknown"; "default": boolean | null };
export type EngineThinkingMetadata = { "requested": boolean | null; "applied": boolean; "effective": boolean | null; "reason": "applied" | "already_fixed" | "model_default" | "unsupported_model" | "fixed_mode" | "unknown_support"; "source": "engine" };
export const CHAT_V2_INPUT_SCHEMA: unknown = {
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "messages": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": false,
        "properties": {
          "role": {
            "enum": [
              "system",
              "user",
              "assistant"
            ]
          },
          "content": {
            "type": "string",
            "minLength": 1,
            "maxLength": 32768
          }
        },
        "required": [
          "role",
          "content"
        ]
      },
      "minItems": 1,
      "maxItems": 128
    },
    "imageArtifactId": {
      "type": "string",
      "pattern": "^artifact_[A-Za-z0-9_-]{20,80}$"
    },
    "maxTokens": {
      "type": "integer",
      "minimum": 1,
      "maximum": 8192,
      "default": 4096
    },
    "temperature": {
      "type": "number",
      "minimum": 0,
      "maximum": 2,
      "default": 0.7
    },
    "topP": {
      "type": "number",
      "exclusiveMinimum": 0,
      "maximum": 1,
      "default": 0.9
    },
    "thinking": {
      "type": "boolean"
    }
  },
  "required": [
    "messages"
  ]
};
export const CHAT_V2_RESULT_SCHEMA: unknown = {
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "kind": {
      "const": "structured"
    },
    "value": {
      "type": "object",
      "additionalProperties": false,
      "required": [
        "text",
        "finishReason",
        "usage",
        "metadata"
      ],
      "properties": {
        "text": {
          "type": "string",
          "maxLength": 1048576
        },
        "reasoning": {
          "type": "string",
          "maxLength": 1048576
        },
        "finishReason": {
          "enum": [
            "stop",
            "length",
            "unknown"
          ]
        },
        "usage": {
          "type": "object",
          "additionalProperties": false,
          "required": [
            "inputTokens",
            "outputTokens",
            "reasoningTokens",
            "answerTokens"
          ],
          "properties": {
            "inputTokens": {
              "anyOf": [
                {
                  "type": "integer",
                  "minimum": 0,
                  "maximum": 4194304
                },
                {
                  "type": "null"
                }
              ]
            },
            "outputTokens": {
              "anyOf": [
                {
                  "type": "integer",
                  "minimum": 0,
                  "maximum": 8192
                },
                {
                  "type": "null"
                }
              ]
            },
            "reasoningTokens": {
              "anyOf": [
                {
                  "type": "integer",
                  "minimum": 0,
                  "maximum": 8192
                },
                {
                  "type": "null"
                }
              ]
            },
            "answerTokens": {
              "anyOf": [
                {
                  "type": "integer",
                  "minimum": 0,
                  "maximum": 8192
                },
                {
                  "type": "null"
                }
              ]
            }
          }
        },
        "metadata": {
          "type": "object",
          "additionalProperties": false,
          "required": [
            "thinking"
          ],
          "properties": {
            "thinking": {
              "type": "object",
              "additionalProperties": false,
              "required": [
                "requested",
                "applied",
                "effective",
                "reason",
                "source"
              ],
              "properties": {
                "requested": {
                  "anyOf": [
                    {
                      "type": "boolean"
                    },
                    {
                      "type": "null"
                    }
                  ]
                },
                "applied": {
                  "type": "boolean"
                },
                "effective": {
                  "anyOf": [
                    {
                      "type": "boolean"
                    },
                    {
                      "type": "null"
                    }
                  ]
                },
                "reason": {
                  "enum": [
                    "applied",
                    "already_fixed",
                    "model_default",
                    "unsupported_model",
                    "fixed_mode",
                    "unknown_support"
                  ]
                },
                "source": {
                  "const": "engine"
                }
              }
            }
          }
        }
      }
    }
  },
  "required": [
    "kind",
    "value"
  ]
};
export const CHAT_V1_RESULT_SCHEMA: unknown = {
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "kind": {
      "const": "structured"
    },
    "value": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "text": {
          "type": "string",
          "minLength": 0,
          "maxLength": 1048576
        }
      },
      "required": [
        "text"
      ]
    }
  },
  "required": [
    "kind",
    "value"
  ]
};
