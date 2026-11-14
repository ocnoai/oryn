import { expect, it } from 'vitest';
import { modelIdentityKey } from '../../shared/model-identity';
it('keys exact revisions and file sets without merging distinct artifacts', () => {
  const model = {
    provider: 'huggingface',
    repository: 'validation/model',
    revision: 'a'.repeat(40),
  };
  expect(modelIdentityKey(model)).not.toBe(
    modelIdentityKey({ ...model, revision: 'b'.repeat(40) }),
  );
  expect(modelIdentityKey({ ...model, files: ['a.safetensors'] })).not.toBe(
    modelIdentityKey({ ...model, files: ['b.safetensors'] }),
  );
  expect(modelIdentityKey({ ...model, files: ['b', 'a'] })).toBe(
    modelIdentityKey({ ...model, files: ['a', 'b'] }),
  );
  expect(modelIdentityKey(model)).not.toBe(modelIdentityKey({ ...model, files: [] }));
});
