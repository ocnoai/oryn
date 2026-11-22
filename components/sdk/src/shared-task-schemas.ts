// Generated from canonical Protocol input schemas; do not hand-edit.
export const sharedTaskSchemas: Record<string, unknown> = {
  "unconditional-image-generation-v1": {
    "type": "object",
    "additionalProperties": false,
    "properties": {
      "width": {
        "type": "integer",
        "minimum": 8,
        "maximum": 1024,
        "multipleOf": 8,
        "default": 256
      },
      "height": {
        "type": "integer",
        "minimum": 8,
        "maximum": 1024,
        "multipleOf": 8,
        "default": 256
      },
      "steps": {
        "type": "integer",
        "minimum": 1,
        "maximum": 1000,
        "default": 50
      },
      "seed": {
        "type": "integer",
        "minimum": 0,
        "maximum": 2147483647,
        "default": 0
      }
    },
    "required": []
  },
  "audio-question-answering-v1": {
    "type": "object",
    "additionalProperties": false,
    "properties": {
      "audioArtifactId": {
        "type": "string",
        "pattern": "^artifact_[A-Za-z0-9_-]{20,80}$"
      },
      "instruction": {
        "type": "string",
        "minLength": 1,
        "maxLength": 2048,
        "pattern": "\\S"
      },
      "maxTokens": {
        "type": "integer",
        "minimum": 1,
        "maximum": 256,
        "default": 128
      }
    },
    "required": [
      "audioArtifactId",
      "instruction"
    ]
  },
  "audio-transformation-v1": {
    "type": "object",
    "additionalProperties": false,
    "properties": {
      "operation": {
        "enum": [
          "enhance",
          "separate",
          "voice-conversion",
          "encode",
          "decode",
          "vocode"
        ]
      },
      "audioArtifactId": {
        "type": "string",
        "pattern": "^artifact_[A-Za-z0-9_-]{20,80}$"
      },
      "referenceAudioArtifactId": {
        "type": "string",
        "pattern": "^artifact_[A-Za-z0-9_-]{20,80}$"
      },
      "codes": {
        "type": "array",
        "items": {
          "type": "array",
          "items": {
            "type": "integer",
            "minimum": 0,
            "maximum": 65535
          },
          "minItems": 1,
          "maxItems": 32
        },
        "minItems": 1,
        "maxItems": 2048
      },
      "spectrogram": {
        "type": "array",
        "items": {
          "type": "array",
          "items": {
            "type": "number",
            "minimum": -1000,
            "maximum": 1000
          },
          "minItems": 1,
          "maxItems": 128
        },
        "minItems": 1,
        "maxItems": 512
      },
      "sampleRate": {
        "type": "integer",
        "minimum": 8000,
        "maximum": 48000
      }
    },
    "required": [
      "operation"
    ]
  },
  "multimodal-generation-v1": {
    "type": "object",
    "additionalProperties": false,
    "properties": {
      "instruction": {
        "type": "string",
        "minLength": 1,
        "maxLength": 2048,
        "pattern": "\\S"
      },
      "imageArtifactId": {
        "type": "string",
        "pattern": "^artifact_[A-Za-z0-9_-]{20,80}$"
      },
      "audioArtifactId": {
        "type": "string",
        "pattern": "^artifact_[A-Za-z0-9_-]{20,80}$"
      },
      "inputMode": {
        "enum": [
          "image",
          "audio",
          "image-and-audio"
        ]
      },
      "outputMode": {
        "enum": [
          "text",
          "audio",
          "text-and-audio"
        ],
        "default": "text"
      },
      "maxTokens": {
        "type": "integer",
        "minimum": 1,
        "maximum": 256,
        "default": 128
      }
    },
    "required": [
      "instruction",
      "inputMode"
    ]
  },
  "image-text-to-video-v1": {
    "type": "object",
    "additionalProperties": false,
    "properties": {
      "imageArtifactId": {
        "type": "string",
        "pattern": "^artifact_[A-Za-z0-9_-]{20,80}$"
      },
      "prompt": {
        "type": "string",
        "minLength": 1,
        "maxLength": 2048,
        "pattern": "\\S"
      },
      "width": {
        "type": "integer",
        "minimum": 64,
        "maximum": 1024,
        "multipleOf": 16,
        "default": 256
      },
      "height": {
        "type": "integer",
        "minimum": 64,
        "maximum": 1024,
        "multipleOf": 16,
        "default": 256
      },
      "frames": {
        "type": "integer",
        "minimum": 1,
        "maximum": 360,
        "default": 17
      },
      "fps": {
        "type": "integer",
        "minimum": 1,
        "maximum": 30,
        "default": 8
      },
      "steps": {
        "type": "integer",
        "minimum": 1,
        "maximum": 100,
        "default": 20
      },
      "guidanceScale": {
        "type": "number",
        "minimum": 0,
        "maximum": 30,
        "default": 6
      },
      "seed": {
        "type": "integer",
        "minimum": 0,
        "maximum": 2147483647,
        "default": 0
      }
    },
    "required": [
      "imageArtifactId",
      "prompt"
    ]
  },
  "video-editing-v1": {
    "type": "object",
    "additionalProperties": false,
    "properties": {
      "videoArtifactId": {
        "type": "string",
        "pattern": "^artifact_[A-Za-z0-9_-]{20,80}$"
      },
      "prompt": {
        "type": "string",
        "minLength": 1,
        "maxLength": 2048,
        "pattern": "\\S"
      },
      "maskVideoArtifactId": {
        "type": "string",
        "pattern": "^artifact_[A-Za-z0-9_-]{20,80}$"
      },
      "operation": {
        "enum": [
          "edit",
          "restore"
        ],
        "default": "edit"
      },
      "startSeconds": {
        "type": "number",
        "minimum": 0,
        "maximum": 30,
        "default": 0
      },
      "durationSeconds": {
        "type": "number",
        "minimum": 0.01,
        "maximum": 30,
        "default": 1
      },
      "width": {
        "type": "integer",
        "minimum": 64,
        "maximum": 1024,
        "multipleOf": 16,
        "default": 256
      },
      "height": {
        "type": "integer",
        "minimum": 64,
        "maximum": 1024,
        "multipleOf": 16,
        "default": 256
      },
      "frames": {
        "type": "integer",
        "minimum": 1,
        "maximum": 129,
        "default": 17
      },
      "fps": {
        "type": "integer",
        "minimum": 1,
        "maximum": 30,
        "default": 8
      },
      "steps": {
        "type": "integer",
        "minimum": 1,
        "maximum": 100,
        "default": 20
      },
      "guidanceScale": {
        "type": "number",
        "minimum": 0,
        "maximum": 30,
        "default": 6
      },
      "seed": {
        "type": "integer",
        "minimum": 0,
        "maximum": 2147483647,
        "default": 0
      }
    },
    "required": [
      "videoArtifactId"
    ]
  },
  "tabular-classification-v1": {
    "type": "object",
    "additionalProperties": false,
    "properties": {
      "columns": {
        "type": "array",
        "items": {
          "type": "object",
          "additionalProperties": false,
          "properties": {
            "name": {
              "type": "string",
              "minLength": 1,
              "maxLength": 128,
              "pattern": "\\S"
            },
            "type": {
              "enum": [
                "numeric",
                "categorical"
              ]
            }
          },
          "required": [
            "name",
            "type"
          ]
        },
        "minItems": 1,
        "maxItems": 32
      },
      "trainingRows": {
        "type": "array",
        "items": {
          "type": "array",
          "items": {
            "anyOf": [
              {
                "type": "number",
                "minimum": -1000000000000,
                "maximum": 1000000000000
              },
              {
                "type": "string",
                "maxLength": 256
              },
              {
                "type": "null"
              }
            ]
          },
          "minItems": 1,
          "maxItems": 32
        },
        "minItems": 2,
        "maxItems": 256
      },
      "queryRows": {
        "type": "array",
        "items": {
          "type": "array",
          "items": {
            "anyOf": [
              {
                "type": "number",
                "minimum": -1000000000000,
                "maximum": 1000000000000
              },
              {
                "type": "string",
                "maxLength": 256
              },
              {
                "type": "null"
              }
            ]
          },
          "minItems": 1,
          "maxItems": 32
        },
        "minItems": 1,
        "maxItems": 64
      },
      "targets": {
        "type": "array",
        "items": {
          "type": "string",
          "minLength": 1,
          "maxLength": 128,
          "pattern": "\\S"
        },
        "minItems": 2,
        "maxItems": 256
      }
    },
    "required": [
      "columns",
      "trainingRows",
      "queryRows",
      "targets"
    ]
  },
  "tabular-regression-v1": {
    "type": "object",
    "additionalProperties": false,
    "properties": {
      "columns": {
        "type": "array",
        "items": {
          "type": "object",
          "additionalProperties": false,
          "properties": {
            "name": {
              "type": "string",
              "minLength": 1,
              "maxLength": 128,
              "pattern": "\\S"
            },
            "type": {
              "enum": [
                "numeric",
                "categorical"
              ]
            }
          },
          "required": [
            "name",
            "type"
          ]
        },
        "minItems": 1,
        "maxItems": 32
      },
      "trainingRows": {
        "type": "array",
        "items": {
          "type": "array",
          "items": {
            "anyOf": [
              {
                "type": "number",
                "minimum": -1000000000000,
                "maximum": 1000000000000
              },
              {
                "type": "string",
                "maxLength": 256
              },
              {
                "type": "null"
              }
            ]
          },
          "minItems": 1,
          "maxItems": 32
        },
        "minItems": 2,
        "maxItems": 256
      },
      "queryRows": {
        "type": "array",
        "items": {
          "type": "array",
          "items": {
            "anyOf": [
              {
                "type": "number",
                "minimum": -1000000000000,
                "maximum": 1000000000000
              },
              {
                "type": "string",
                "maxLength": 256
              },
              {
                "type": "null"
              }
            ]
          },
          "minItems": 1,
          "maxItems": 32
        },
        "minItems": 1,
        "maxItems": 64
      },
      "targets": {
        "type": "array",
        "items": {
          "type": "number",
          "minimum": -1000000000000,
          "maximum": 1000000000000
        },
        "minItems": 2,
        "maxItems": 256
      }
    },
    "required": [
      "columns",
      "trainingRows",
      "queryRows",
      "targets"
    ]
  },
  "time-series-forecasting-v1": {
    "type": "object",
    "additionalProperties": false,
    "properties": {
      "context": {
        "type": "array",
        "items": {
          "type": "number",
          "minimum": -1000000000000,
          "maximum": 1000000000000
        },
        "minItems": 2,
        "maxItems": 2048
      },
      "horizon": {
        "type": "integer",
        "minimum": 1,
        "maximum": 512,
        "default": 16
      },
      "quantiles": {
        "type": "array",
        "items": {
          "type": "number",
          "minimum": 0.01,
          "maximum": 0.99
        },
        "minItems": 1,
        "maxItems": 19,
        "uniqueItems": true
      },
      "timestamps": {
        "type": "array",
        "items": {
          "type": "integer",
          "minimum": 0,
          "maximum": 8640000000000000
        },
        "minItems": 2,
        "maxItems": 2048
      },
      "frequencyMilliseconds": {
        "type": "integer",
        "minimum": 1,
        "maximum": 31536000000
      },
      "pastCovariates": {
        "type": "array",
        "items": {
          "type": "array",
          "items": {
            "type": "number",
            "minimum": -1000000000000,
            "maximum": 1000000000000
          },
          "minItems": 1,
          "maxItems": 16
        },
        "minItems": 2,
        "maxItems": 2048
      },
      "futureCovariates": {
        "type": "array",
        "items": {
          "type": "array",
          "items": {
            "type": "number",
            "minimum": -1000000000000,
            "maximum": 1000000000000
          },
          "minItems": 1,
          "maxItems": 16
        },
        "minItems": 1,
        "maxItems": 512
      }
    },
    "required": [
      "context"
    ]
  },
  "pose-estimation-v1": {
    "type": "object",
    "additionalProperties": false,
    "properties": {
      "imageArtifactId": {
        "type": "string",
        "pattern": "^artifact_[A-Za-z0-9_-]{20,80}$"
      },
      "boxes": {
        "type": "array",
        "items": {
          "type": "array",
          "items": {
            "type": "number",
            "minimum": 0,
            "maximum": 1
          },
          "minItems": 4,
          "maxItems": 4
        },
        "minItems": 1,
        "maxItems": 8
      }
    },
    "required": [
      "imageArtifactId",
      "boxes"
    ]
  },
  "visual-document-retrieval-v1": {
    "type": "object",
    "additionalProperties": false,
    "properties": {
      "imageArtifactId": {
        "type": "string",
        "pattern": "^artifact_[A-Za-z0-9_-]{20,80}$"
      },
      "queries": {
        "type": "array",
        "items": {
          "type": "string",
          "minLength": 1,
          "maxLength": 256,
          "pattern": "\\S"
        },
        "minItems": 1,
        "maxItems": 2
      }
    },
    "required": [
      "imageArtifactId",
      "queries"
    ]
  },
  "table-question-answering-v1": {
    "type": "object",
    "additionalProperties": false,
    "properties": {
      "question": {
        "type": "string",
        "minLength": 1,
        "maxLength": 2048,
        "pattern": "\\S"
      },
      "columns": {
        "type": "array",
        "items": {
          "type": "string",
          "minLength": 1,
          "maxLength": 128,
          "pattern": "\\S"
        },
        "minItems": 1,
        "maxItems": 32
      },
      "rows": {
        "type": "array",
        "items": {
          "type": "array",
          "items": {
            "type": "string",
            "maxLength": 1024
          },
          "minItems": 1,
          "maxItems": 32
        },
        "minItems": 1,
        "maxItems": 128
      }
    },
    "required": [
      "question",
      "columns",
      "rows"
    ]
  },
  "document-question-answering-v1": {
    "type": "object",
    "additionalProperties": false,
    "properties": {
      "imageArtifactId": {
        "type": "string",
        "pattern": "^artifact_[A-Za-z0-9_-]{20,80}$"
      },
      "question": {
        "type": "string",
        "minLength": 1,
        "maxLength": 2048,
        "pattern": "\\S"
      },
      "words": {
        "type": "array",
        "items": {
          "type": "string",
          "minLength": 1,
          "maxLength": 128,
          "pattern": "\\S"
        },
        "minItems": 1,
        "maxItems": 512
      },
      "boxes": {
        "type": "array",
        "items": {
          "type": "array",
          "items": {
            "type": "number",
            "minimum": 0,
            "maximum": 1
          },
          "minItems": 4,
          "maxItems": 4
        },
        "minItems": 1,
        "maxItems": 512
      }
    },
    "required": [
      "imageArtifactId",
      "question",
      "words",
      "boxes"
    ]
  },
  "keypoint-detection-v1": {
    "type": "object",
    "additionalProperties": false,
    "properties": {
      "imageArtifactId": {
        "type": "string",
        "pattern": "^artifact_[A-Za-z0-9_-]{20,80}$"
      },
      "threshold": {
        "type": "number",
        "minimum": 0,
        "maximum": 1,
        "default": 0.005
      },
      "maxKeypoints": {
        "type": "integer",
        "minimum": 1,
        "maximum": 1024,
        "default": 256
      }
    },
    "required": [
      "imageArtifactId"
    ]
  },
  "video-classification-v1": {
    "type": "object",
    "additionalProperties": false,
    "properties": {
      "videoArtifactId": {
        "type": "string",
        "pattern": "^artifact_[A-Za-z0-9_-]{20,80}$"
      },
      "startSeconds": {
        "type": "number",
        "minimum": 0,
        "maximum": 30
      },
      "durationSeconds": {
        "type": "number",
        "minimum": 0,
        "maximum": 30,
        "exclusiveMinimum": 0
      },
      "topK": {
        "type": "integer",
        "minimum": 1,
        "maximum": 20,
        "default": 5
      }
    },
    "required": [
      "videoArtifactId"
    ]
  },
  "video-question-answering-v1": {
    "type": "object",
    "additionalProperties": false,
    "properties": {
      "videoArtifactId": {
        "type": "string",
        "pattern": "^artifact_[A-Za-z0-9_-]{20,80}$"
      },
      "startSeconds": {
        "type": "number",
        "minimum": 0,
        "maximum": 30
      },
      "durationSeconds": {
        "type": "number",
        "minimum": 0,
        "maximum": 30,
        "exclusiveMinimum": 0
      },
      "question": {
        "type": "string",
        "minLength": 1,
        "maxLength": 2048,
        "pattern": "\\S"
      },
      "maxTokens": {
        "type": "integer",
        "minimum": 1,
        "maximum": 256,
        "default": 128
      }
    },
    "required": [
      "videoArtifactId",
      "question"
    ]
  },
  "voice-activity-detection-v1": {
    "type": "object",
    "additionalProperties": false,
    "properties": {
      "audioArtifactId": {
        "type": "string",
        "pattern": "^artifact_[A-Za-z0-9_-]{20,80}$"
      },
      "threshold": {
        "type": "number",
        "minimum": 0.01,
        "maximum": 0.99,
        "default": 0.5
      },
      "minSpeechSeconds": {
        "type": "number",
        "minimum": 0.032,
        "maximum": 2,
        "default": 0.25
      },
      "minSilenceSeconds": {
        "type": "number",
        "minimum": 0.032,
        "maximum": 2,
        "default": 0.1
      }
    },
    "required": [
      "audioArtifactId"
    ]
  },
  "image-editing-v1": {
    "type": "object",
    "additionalProperties": false,
    "required": [
      "imageArtifactId",
      "prompt"
    ],
    "properties": {
      "imageArtifactId": {
        "type": "string",
        "pattern": "^artifact_[A-Za-z0-9_-]{20,80}$"
      },
      "prompt": {
        "type": "string",
        "minLength": 1,
        "maxLength": 4096
      },
      "negativePrompt": {
        "type": "string",
        "maxLength": 4096
      },
      "steps": {
        "type": "integer",
        "minimum": 1,
        "maximum": 50
      },
      "guidanceScale": {
        "type": "number",
        "minimum": 0,
        "maximum": 10
      },
      "seed": {
        "type": "integer",
        "minimum": 0,
        "maximum": 4294967295
      },
      "width": {
        "type": "integer",
        "minimum": 64,
        "maximum": 1024,
        "multipleOf": 16
      },
      "height": {
        "type": "integer",
        "minimum": 64,
        "maximum": 1024,
        "multipleOf": 16
      },
      "referenceImageArtifactIds": {
        "type": "array",
        "maxItems": 2,
        "uniqueItems": true,
        "items": {
          "type": "string",
          "pattern": "^artifact_[A-Za-z0-9_-]{20,80}$"
        }
      }
    }
  },
  "image-layers-v1": {
    "type": "object",
    "additionalProperties": false,
    "required": [
      "imageArtifactId",
      "prompt"
    ],
    "properties": {
      "imageArtifactId": {
        "type": "string",
        "pattern": "^artifact_[A-Za-z0-9_-]{20,80}$"
      },
      "prompt": {
        "type": "string",
        "minLength": 1,
        "maxLength": 4096
      },
      "negativePrompt": {
        "type": "string",
        "maxLength": 4096
      },
      "steps": {
        "type": "integer",
        "minimum": 1,
        "maximum": 50
      },
      "guidanceScale": {
        "type": "number",
        "minimum": 0,
        "maximum": 10
      },
      "seed": {
        "type": "integer",
        "minimum": 0,
        "maximum": 4294967295
      },
      "resolution": {
        "type": "integer",
        "enum": [
          640,
          1024
        ]
      },
      "layers": {
        "type": "integer",
        "minimum": 1,
        "maximum": 4
      }
    }
  },
  "zero-shot-classification-v1": {
    "type": "object",
    "additionalProperties": false,
    "properties": {
      "text": {
        "type": "string",
        "minLength": 1,
        "maxLength": 8192,
        "pattern": "\\S"
      },
      "labels": {
        "type": "array",
        "minItems": 2,
        "maxItems": 32,
        "uniqueItems": true,
        "items": {
          "type": "string",
          "minLength": 1,
          "maxLength": 128,
          "pattern": "\\S"
        }
      },
      "multiLabel": {
        "type": "boolean"
      }
    },
    "required": [
      "text",
      "labels"
    ]
  },
  "image-embeddings-v1": {
    "type": "object",
    "additionalProperties": false,
    "properties": {
      "imageArtifactId": {
        "type": "string",
        "pattern": "^artifact_[A-Za-z0-9_-]{20,80}$"
      },
      "normalize": {
        "type": "boolean"
      }
    },
    "required": [
      "imageArtifactId"
    ]
  },
  "image-segmentation-v1": {
    "type": "object",
    "additionalProperties": false,
    "properties": {
      "imageArtifactId": {
        "type": "string",
        "pattern": "^artifact_[A-Za-z0-9_-]{20,80}$"
      }
    },
    "required": [
      "imageArtifactId"
    ]
  },
  "mask-generation-v1": {
    "type": "object",
    "additionalProperties": false,
    "properties": {
      "imageArtifactId": {
        "type": "string",
        "pattern": "^artifact_[A-Za-z0-9_-]{20,80}$"
      },
      "points": {
        "type": "array",
        "minItems": 1,
        "maxItems": 16,
        "items": {
          "type": "object",
          "additionalProperties": false,
          "properties": {
            "x": {
              "type": "number",
              "minimum": 0,
              "maximum": 1
            },
            "y": {
              "type": "number",
              "minimum": 0,
              "maximum": 1
            },
            "label": {
              "enum": [
                0,
                1
              ],
              "type": "integer"
            }
          },
          "required": [
            "x",
            "y",
            "label"
          ]
        },
        "contains": {
          "type": "object",
          "properties": {
            "label": {
              "const": 1
            }
          },
          "required": [
            "label"
          ]
        },
        "minContains": 1
      }
    },
    "required": [
      "imageArtifactId",
      "points"
    ]
  },
  "zero-shot-object-detection-v1": {
    "type": "object",
    "additionalProperties": false,
    "properties": {
      "imageArtifactId": {
        "type": "string",
        "pattern": "^artifact_[A-Za-z0-9_-]{20,80}$"
      },
      "labels": {
        "type": "array",
        "minItems": 2,
        "maxItems": 32,
        "uniqueItems": true,
        "items": {
          "type": "string",
          "minLength": 1,
          "maxLength": 128,
          "pattern": "\\S"
        }
      },
      "threshold": {
        "type": "number",
        "minimum": 0,
        "maximum": 1
      }
    },
    "required": [
      "imageArtifactId",
      "labels"
    ]
  },
  "image-to-text-v1": {
    "type": "object",
    "additionalProperties": false,
    "properties": {
      "imageArtifactId": {
        "type": "string",
        "pattern": "^artifact_[A-Za-z0-9_-]{20,80}$"
      },
      "maxTokens": {
        "type": "integer",
        "minimum": 1,
        "maximum": 256
      }
    },
    "required": [
      "imageArtifactId"
    ]
  },
  "visual-question-answering-v1": {
    "type": "object",
    "additionalProperties": false,
    "properties": {
      "imageArtifactId": {
        "type": "string",
        "pattern": "^artifact_[A-Za-z0-9_-]{20,80}$"
      },
      "question": {
        "type": "string",
        "minLength": 1,
        "maxLength": 2048,
        "pattern": "\\S"
      },
      "maxTokens": {
        "type": "integer",
        "minimum": 1,
        "maximum": 256
      }
    },
    "required": [
      "imageArtifactId",
      "question"
    ]
  }
};
