/** Explicit development acceptance through the real isolated workspace bridge. */
import { createHash } from 'node:crypto';
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { basename, resolve } from 'node:path';
import { createServer } from 'node:http';
import { setTimeout as delay } from 'node:timers/promises';
import { readFileSync as readArchiveFileSync } from 'original-fs';
import type { ResolvedModelRef, TaskRequirements } from '@ocno/oryn-protocol';
import type { BrowserWindowController } from '../browser-window';
import type { EngineSupervisor } from '../engine-supervisor';
import type { ModelWorkspaceStore } from './model-workspace-store';
import type { ModelResolver } from '../model-resolution/model-resolver';
import { createOrynModelWorkspaceURL } from '../../shared/browser-urls';
import { taskContractForProfile } from './model-workspace-eligibility';

type AcceptanceCase = {
  model: ResolvedModelRef;
  requirements: TaskRequirements;
  recommendation?: { snapshotDigest: string; engines: string[] };
  input: Record<string, unknown>;
  bindings?: { artifactId: string; path: string; mimeType: string }[];
  expected?: Record<string, unknown>;
  expectedFoundation?: string;
  cancelInWorkspace?: boolean;
  cancellationStage?: string;
  selectionIntent?: 'source-checkpoint' | 'preferred-variant' | 'exact-artifact';
};

export async function runWorkspaceAcceptance(
  controller: BrowserWindowController,
  engine: EngineSupervisor,
  workspaces: ModelWorkspaceStore,
  resolver: ModelResolver,
  casesPath: string,
  reportPath: string,
): Promise<{ passed: boolean; cases: number; failures: number }> {
  if (process.env.LOCAL_AI_SMOKE_TEST !== '1') throw new Error('Acceptance requires smoke mode');
  const website = process.env.ORYN_ACCEPTANCE_SURFACE === 'website';
  const server = website
    ? createServer((_request, response) => {
        response.writeHead(200, { 'Content-Type': 'text/html', 'Cache-Control': 'no-store' });
        response.end(
          '<!doctype html><title>Local AI website acceptance</title><main>Local AI test fixture</main>',
        );
      })
    : undefined;
  if (server) await new Promise<void>((done) => server.listen(0, '127.0.0.1', done));
  try {
    const address = server?.address();
    const websiteURL =
      address && typeof address !== 'string' ? `http://127.0.0.1:${address.port}` : undefined;
    return await runAcceptance(
      controller,
      engine,
      workspaces,
      resolver,
      casesPath,
      reportPath,
      websiteURL,
    );
  } finally {
    if (server) await new Promise<void>((done) => server.close(() => done()));
  }
}

async function runAcceptance(
  controller: BrowserWindowController,
  engine: EngineSupervisor,
  workspaces: ModelWorkspaceStore,
  resolver: ModelResolver,
  casesPath: string,
  reportPath: string,
  websiteURL?: string,
): Promise<{ passed: boolean; cases: number; failures: number }> {
  await engine.start();
  const raw = readFileSync(resolve(casesPath));
  if (raw.byteLength > 1024 * 1024) throw new Error('Acceptance cases exceed bound');
  const cases = JSON.parse(raw.toString()) as AcceptanceCase[];
  if (!Array.isArray(cases) || cases.length < 1 || cases.length > 100)
    throw new Error('Acceptance case count exceeds bound');
  const manifestPath = resolve(process.resourcesPath, 'engine/release-manifest.json');
  const applicationPath = resolve(process.resourcesPath, 'app.asar');
  const packageBuild = existsSync(manifestPath)
    ? {
        releaseManifest: JSON.parse(readFileSync(manifestPath, 'utf8')),
        releaseManifestSha256: createHash('sha256')
          .update(readFileSync(manifestPath))
          .digest('hex'),
        browserAsarSha256: createHash('sha256')
          .update(readArchiveFileSync(applicationPath))
          .digest('hex'),
      }
    : { qualification: 'development source invocation' };
  const results: Record<string, unknown>[] = [];
  const save = () =>
    writeFileSync(
      resolve(reportPath),
      `${JSON.stringify(
        {
          schemaVersion: 1,
          packageBuild,
          qualification: websiteURL
            ? 'Browser ordinary website bridge'
            : 'Browser workspace bridge',
          casesSha256: createHash('sha256').update(raw).digest('hex'),
          results,
        },
        null,
        2,
      )}\n`,
    );
  for (const test of cases) {
    const row: Record<string, unknown> = { case: test, status: 'running', samples: [] };
    results.push(row);
    let tabOpen = false;
    let preparedModel: ResolvedModelRef | undefined;
    try {
      let model = test.model;
      let recommendation = test.recommendation;
      if (test.selectionIntent) {
        const inventory = await engine.getModelInventory();
        const resolution = await resolver.resolve(
          {
            role: 'chat',
            requirements: test.requirements,
            selection: {
              version: 1,
              intent: test.selectionIntent,
              ...(test.selectionIntent === 'source-checkpoint'
                ? {
                    sourceCheckpoint: {
                      provider: model.provider,
                      repository: model.repository,
                      revision: model.revision,
                    },
                  }
                : {}),
            },
            ...(test.selectionIntent === 'source-checkpoint' ? {} : { preferredModel: model }),
            substitution: { automatic: 'none' },
          },
          { origin: 'https://sprint59.example', inventory: inventory.models },
        );
        row.resolution = resolution;
        model = resolution.selected.model;
        if (model.repository !== test.model.repository || model.revision !== test.model.revision)
          throw new Error('RESOLVER_CHANGED_EXACT_IDENTITY');
        recommendation = resolution.selected.recommendation
          ? {
              snapshotDigest: resolution.selected.recommendation.snapshotDigest,
              engines: resolution.selected.recommendation.engines,
            }
          : undefined;
      }
      const profile = await engine.inspectModel(model, test.requirements, recommendation);
      row.inspection = profile;
      const taskContract = taskContractForProfile(profile);
      if (!profile.executable || !taskContract) throw new Error('INSPECTION_BLOCKED');
      // Prepare the selected offer just as BrowserModelSessions does before attachment.
      await engine.installModel(
        `install_sprint59_${results.length}`,
        model,
        test.requirements,
        recommendation,
      );
      preparedModel = profile.model;
      const binding = workspaces.create({
        model: profile.model,
        selectedFiles:
          profile.selectedFiles?.map((file) => file.name).sort() ?? profile.model.files,
        taskContract,
        requirements: test.requirements,
        trustMode: 'user-trusted',
        ...(profile.runtimePack
          ? {
              runtimePack: {
                packId: profile.runtimePack.packId,
                packVersion: profile.runtimePack.packVersion,
              },
            }
          : {}),
      });
      const inputs = (test.bindings ?? []).map((item) => {
        const bytes = readFileSync(resolve(item.path));
        if (bytes.byteLength > 64 * 1024 * 1024) throw new Error('Fixture exceeds bound');
        return {
          ...item,
          name: basename(item.path),
          data: bytes.toString('base64'),
          sha256: createHash('sha256').update(bytes).digest('hex'),
        };
      });
      row.fixtures = inputs.map(({ artifactId, path, mimeType, sha256 }) => ({
        artifactId,
        path,
        mimeType,
        sha256,
      }));
      const audioExpected = test.expected?.audioEnhancement as Record<string, unknown> | undefined;
      const audioReferences: Record<string, string> = {};
      const audioReferenceHashes: Record<string, string> = {};
      for (const key of ['referencePath', 'cleanPath', 'noisyPath']) {
        if (audioExpected?.[key] !== undefined) {
          if (typeof audioExpected[key] !== 'string') throw new Error('Invalid audio reference');
          const bytes = readFileSync(resolve(audioExpected[key]));
          if (bytes.byteLength > 6 * 1024 * 1024) throw new Error('Audio reference exceeds bound');
          audioReferences[key] = bytes.toString('base64');
          audioReferenceHashes[key] = createHash('sha256').update(bytes).digest('hex');
        }
      }
      if (audioExpected) row.audioReferenceHashes = audioReferenceHashes;
      for (let repeat = 0; repeat < 2; repeat++) {
        controller.createTab(websiteURL ?? createOrynModelWorkspaceURL(binding.reference), true);
        tabOpen = true;
        const contents = controller.activeWebsiteWebContents;
        if (contents.isLoading())
          await new Promise<void>((done) => contents.once('did-finish-load', () => done()));
        if (websiteURL) {
          await contents.executeJavaScript(`
            globalThis.acceptanceSession = null;
            void globalThis.localAI.requestModel('acceptance_${results.length}_${repeat}', ${JSON.stringify(
              {
                role: test.requirements.task,
                preferredModel: {
                  ...profile.model,
                  files: profile.selectedFiles?.map((f) => f.name).sort(),
                },
                requirements: test.requirements,
                substitution: { automatic: 'none' },
              },
            )}).then(value => { globalThis.acceptanceSession = value; })
              .catch(error => { globalThis.acceptanceSession = {ok:false,error:{code:String(error)}}; });
          `);
          const deadline = Date.now() + 180_000;
          while (!(await contents.executeJavaScript('Boolean(globalThis.acceptanceSession)'))) {
            if (Date.now() > deadline) throw new Error('WEBSITE_SESSION_TIMEOUT');
            await controller.trustedWebContents.executeJavaScript(`(() => {
              const dialog = document.querySelector('[aria-label="Local AI permission"]');
              Array.from(dialog?.querySelectorAll('button') ?? []).find(button =>
                ['Allow once','Trust and allow once','Use anyway once'].includes(button.textContent?.trim() ?? '')
              )?.click();
            })()`);
            await delay(100);
          }
        }
        const sample: unknown = await contents.executeJavaScript(`(async () => {
          const website = ${Boolean(websiteURL)};
          const session = globalThis.acceptanceSession;
          if (website && (!session?.ok || 'runtime' in session.value)) throw new Error('WEBSITE_SESSION_INVALID:' + JSON.stringify(session));
          if (website && (session.value.resolution.resolvedModel.repository !== ${JSON.stringify(profile.model.repository)} || session.value.resolution.resolvedModel.revision !== ${JSON.stringify(profile.model.revision)})) throw new Error('WEBSITE_IDENTITY_MISMATCH');
          const bridge = globalThis.localAI;
          const api = website ? {
            ...bridge,
            getState: async () => ({status:'ready', taskContract:session.value.taskProfile?.contract.id}),
            invoke: input => bridge.invokeJob({sessionId:session.value.sessionId,input}),
            cancel: jobId => bridge.cancelJob(jobId),
            onJobEvent: listener => bridge.onJobEvent(event => {if(event.sessionId===session.value.sessionId) listener(event);}),
          } : globalThis.orynWorkspace;
          if (typeof globalThis.process !== 'undefined' || typeof globalThis.require !== 'undefined'
            || (website ? typeof globalThis.orynWorkspace !== 'undefined' : typeof globalThis.localAI !== 'undefined') || !api) throw new Error('WORKSPACE_ISOLATION_FAILED');
          const state = await api.getState();
          if (state.status !== 'ready' || state.taskContract !== ${JSON.stringify(taskContract)})
            throw new Error('WORKSPACE_NOT_READY:' + JSON.stringify(state));
          if (!website && ${JSON.stringify(test.expectedFoundation ?? null)} && state.runtimeSelection?.foundationId !== ${JSON.stringify(test.expectedFoundation ?? null)}) throw new Error('SELECTED_RUNTIME_MISMATCH');
          const input = ${JSON.stringify(test.input)};
          const fixtures = ${JSON.stringify(inputs)};
          const owned = [];
          for (const fixture of fixtures) {
            const bytes = Uint8Array.from(atob(fixture.data), c => c.charCodeAt(0));
            const begin = await api.beginArtifactCreate({name: fixture.name, mimeType: fixture.mimeType, sizeBytes: bytes.length});
            if (!begin.ok) throw new Error(begin.error.code);
            for (let offset=0; offset<bytes.length; offset+=256*1024) {
              const sent = await api.appendArtifactChunk(begin.value.uploadId, bytes.slice(offset, offset+256*1024).buffer);
              if (!sent.ok) throw new Error(sent.error.code);
            }
            const end = await api.finishArtifactCreate(begin.value.uploadId);
            if (!end.ok) throw new Error(end.error.code);
            owned.push(end.value.id);
            for (const key of Object.keys(input)) if (input[key] === fixture.artifactId) input[key] = end.value.id;
          }
          const started = performance.now();
          let stop, timer, textDeltaCount = 0;
          const terminal = new Promise(resolve => {
            timer = setTimeout(() => {stop?.(); resolve({event:'timeout'});}, 600000);
            stop = api.onJobEvent(event => {
              if (event.event === 'job.output.delta' && event.data?.channel === 'text') textDeltaCount++;
              if (['job.completed','job.failed','job.cancelled'].includes(event.event)) {
                clearTimeout(timer); stop(); resolve(event);
              }
            });
          });
          const invalid = await api.invoke({...input, backend:'cpu'});
          if (invalid.ok || invalid.error.code !== (website ? 'JOB_INPUT_INVALID' : 'MODEL_WORKSPACE_INPUT_INVALID')) throw new Error('RUNTIME_FIELD_REJECTION_MISMATCH:' + JSON.stringify(invalid));
          const invocation = await api.invoke(input);
          if (!invocation.ok) {clearTimeout(timer); stop(); throw new Error(invocation.error.code);}
          const event = await terminal;
          if (event.event !== 'job.completed') throw new Error(JSON.stringify(event));
          const output = event.data.result;
          const expected = ${JSON.stringify(test.expected ?? {})};
          let audioMetrics = null;
          if (expected.audioEnhancement) {
            const references = ${JSON.stringify(audioReferences)};
            const wav = buffer => {
              const view = new DataView(buffer);
              const tag = offset => String.fromCharCode(...new Uint8Array(buffer, offset, 4));
              if(view.byteLength < 44 || tag(0)!=='RIFF' || tag(8)!=='WAVE') throw new Error('AUDIO_HEADER');
              let channels, rate, raw;
              for(let offset=12;offset+8<=view.byteLength;) {
                const size=view.getUint32(offset+4,true), start=offset+8;
                if(start+size>view.byteLength) throw new Error('AUDIO_TRUNCATED');
                if(tag(offset)==='fmt ') {
                  if(size<16 || view.getUint16(start,true)!==1 || view.getUint16(start+14,true)!==16) throw new Error('AUDIO_PCM');
                  channels=view.getUint16(start+2,true);rate=view.getUint32(start+4,true);
                }
                if(tag(offset)==='data') raw=new DataView(buffer,start,size);
                offset=start+size+(size%2);
              }
              if(rate!==48000 || ![1,2].includes(channels) || !raw || raw.byteLength%(channels*2)) throw new Error('AUDIO_GEOMETRY');
              const samples=Array.from({length:raw.byteLength/2},(_,i)=>raw.getInt16(i*2,true)/32768);
              return {samples,channels,frames:samples.length/channels};
            };
            const authorized=await api.getArtifactUrl(output.artifact.id);
            if(!authorized.ok) throw new Error('AUDIO_ARTIFACT_AUTHORITY');
            const response=await fetch(authorized.value);
            if(!response.ok) throw new Error('AUDIO_ARTIFACT_READ');
            const actual=wav(await response.arrayBuffer());
            const ref=key=>wav(Uint8Array.from(atob(references[key]),c=>c.charCodeAt(0)).buffer);
            const reference=ref('referencePath');
            if(actual.frames!==reference.frames || actual.channels!==reference.channels) throw new Error('AUDIO_GEOMETRY');
            let maxError=0;
            for(let i=0;i<actual.samples.length;i++) maxError=Math.max(maxError,Math.abs(actual.samples[i]-reference.samples[i]));
            if(maxError>2/32768) throw new Error('AUDIO_REFERENCE_MISMATCH');
            audioMetrics={frames:actual.frames,channels:actual.channels,maxReferenceError:maxError};
            if(references.cleanPath) {
              const clean=ref('cleanPath'),noisy=ref('noisyPath');
              if(clean.frames!==actual.frames || noisy.frames!==actual.frames || clean.channels!==actual.channels || noisy.channels!==actual.channels) throw new Error('AUDIO_GEOMETRY');
              const sdr=value=>{
                let dot=0,energy=0;for(let i=0;i<value.length;i++){dot+=clean.samples[i]*value[i];energy+=clean.samples[i]**2;}
                const scale=dot/energy;let signal=0,error=0;
                for(let i=0;i<value.length;i++){const target=clean.samples[i]*scale;signal+=target**2;error+=(value[i]-target)**2;}
                return 10*Math.log10(signal/(error+1e-15));
              };
              const gain=sdr(actual.samples)-sdr(noisy.samples);
              if(!Number.isFinite(gain)||gain<expected.audioEnhancement.minimumSDRGain) throw new Error('AUDIO_QUALITY_FAILED');
              audioMetrics.sdrGain=gain;
            } else if(actual.samples.some(value=>value!==0)) throw new Error('AUDIO_SILENCE_FAILED');
            owned.push(output.artifact.id);
          }
          const text = String(output?.value?.text ?? output?.text ?? '').toLowerCase();
          if (expected.textSentenceEnd && !/[.!?]$/.test(text.trim())) throw new Error('SEMANTIC_SENTENCE_TRUNCATED');
          if (expected.textContainsAll?.some(t => !text.includes(t.toLowerCase()))) throw new Error('SEMANTIC_REQUIRED_FACT_MISSING:' + text.slice(0,4096));
          if (expected.textExcludes?.some(t => text.includes(t.toLowerCase()))) throw new Error('SEMANTIC_UNSUPPORTED_FACT:' + text.slice(0,4096));
          if (expected.textContains && !text.includes(expected.textContains.toLowerCase())) throw new Error('TEXT_MISMATCH:' + text.slice(0,4096));
          if (expected.textContainsAny && !expected.textContainsAny.some(t => text.includes(t.toLowerCase()))) throw new Error('TEXT_MISMATCH:' + text.slice(0,4096));
          if (expected.textEquals !== undefined && text.trim() !== expected.textEquals.toLowerCase()) throw new Error('TEXT_MISMATCH:' + text.slice(0,4096));
          if (expected.answerEquals && output.value.answer.toLowerCase().trim() !== expected.answerEquals.toLowerCase()) throw new Error('ANSWER_MISMATCH');
          if (expected.topLabel && output.value.predictions[0].label !== expected.topLabel) throw new Error('LABEL_MISMATCH');
          let forecastMAE = null;
          let tabularMAE = null;
          let tabularAccuracy = null;
          if (expected.tabularLabels !== undefined) {
            const target = expected.tabularLabels, predictions = output.value?.predictions;
            if (output.kind !== 'structured' || !Array.isArray(target) || !target.length
              || !target.every(v => typeof v === 'string' && v.length > 0)
              || !Array.isArray(predictions) || predictions.length !== target.length
              || !predictions.every(v => v && typeof v.label === 'string'))
              throw new Error('TABULAR_ACCEPTANCE_INVALID');
            if (!predictions.every((v,i) => v.label === target[i])) throw new Error('SEMANTIC_TABULAR_FAILED');
            tabularAccuracy = 1;
          }
          if (expected.tabularValues !== undefined) {
            const target = expected.tabularValues, predictions = output.value?.predictions;
            if (output.kind !== 'structured' || !Array.isArray(target) || !target.length
              || !Array.isArray(predictions) || predictions.length !== target.length
              || !target.every(v => typeof v === 'number' && Number.isFinite(v))
              || !predictions.every(v => typeof v === 'number' && Number.isFinite(v))
              || typeof expected.maximumTabularMAE !== 'number'
              || !Number.isFinite(expected.maximumTabularMAE) || expected.maximumTabularMAE < 0)
              throw new Error('TABULAR_ACCEPTANCE_INVALID');
            tabularMAE = predictions.reduce((sum,v,i) => sum + Math.abs(v-target[i]),0)/target.length;
            if (tabularMAE > expected.maximumTabularMAE) throw new Error('SEMANTIC_TABULAR_FAILED');
          }
          if (expected.forecastValues !== undefined) {
            const target = expected.forecastValues, predictions = output.value?.predictions;
            if (output.kind !== 'structured' || !Array.isArray(target) || !target.length
              || !Array.isArray(predictions) || predictions.length !== target.length
              || !target.every(v => typeof v === 'number' && Number.isFinite(v))
              || !predictions.every(v => typeof v === 'number' && Number.isFinite(v))
              || typeof expected.maximumForecastMAE !== 'number'
              || !Number.isFinite(expected.maximumForecastMAE) || expected.maximumForecastMAE < 0)
              throw new Error('FORECAST_ACCEPTANCE_INVALID');
            forecastMAE = predictions.reduce((sum,v,i) => sum + Math.abs(v-target[i]),0)/target.length;
            if (forecastMAE > expected.maximumForecastMAE) throw new Error('SEMANTIC_FORECAST_FAILED');
          }
          if (expected.topToken && output.value.predictions[0].token.toLowerCase() !== expected.topToken.toLowerCase()) throw new Error('TOKEN_MISMATCH');
          if (expected.minimumKeypoints && output.value.keypoints.length < expected.minimumKeypoints) throw new Error('KEYPOINT_MISMATCH');
          if (expected.topDocumentIndex !== undefined && output.value.results[0].index !== expected.topDocumentIndex) throw new Error('RANKING_MISMATCH');
          if (expected.wordSpan && JSON.stringify([output.value.wordStart, output.value.wordEnd]) !== JSON.stringify(expected.wordSpan)) throw new Error('DOCUMENT_SPAN_MISMATCH');
          if (expected.speechIntervals !== undefined) {
            const {segments,durationSeconds}=output.value;
            if(Boolean(segments.length)!==expected.speechIntervals || segments.some((s,i)=>s.start<0||s.end<=s.start||s.end>durationSeconds||(i&&segments[i-1].end>s.start))) throw new Error('SPEECH_INTERVALS_MISMATCH');
          }
          if (expected.segmentationLabel || expected.maskPixels) {
            if(output.kind!=='multi'||output.outputs.length!==2) throw new Error('MASK_RESULT_INCOMPLETE');
            const metadata=output.outputs[1].value;
            if(expected.segmentationLabel && !metadata.segments.some(s=>s.label===expected.segmentationLabel&&s.pixelCount>=100)) throw new Error('SEGMENTATION_MISMATCH');
            if(expected.maskPixels) {
              const authorized=await api.getArtifactUrl(output.outputs[0].artifact.id);
              if(!authorized.ok) throw new Error(authorized.error.code);
              const image=new Image();image.crossOrigin='anonymous';
              await new Promise((resolve,reject)=>{image.onload=resolve;image.onerror=()=>reject(new Error('MASK_UNREADABLE'));image.src=authorized.value;});
              const canvas=document.createElement('canvas');canvas.width=image.width;canvas.height=image.height;
              const context=canvas.getContext('2d');context.drawImage(image,0,0);
              if(image.width!==metadata.width||image.height!==metadata.height||expected.maskPixels.some(([x,y,id])=>context.getImageData(x,y,1,1).data[0]!==id)) throw new Error('MASK_PIXELS_MISMATCH');
            }
          }
          if(expected.detections) for(const target of expected.detections) {
            const overlap=d=>{const a=d.box,b=target.box;const intersection=Math.max(0,Math.min(a[2],b[2])-Math.max(a[0],b[0]))*Math.max(0,Math.min(a[3],b[3])-Math.max(a[1],b[1]));return intersection/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection);};
            if(!output.value.detections.some(d=>d.labelId===target.labelId&&d.score>=0.5&&overlap(d)>=0.7)) throw new Error('DETECTION_MISMATCH');
          }
          if (expected.embeddingSimilarityMargin) {
            const vectors = output.value.embeddings;
            const cosine = (a,b) => a.reduce((n,x,i) => n+x*b[i],0) / Math.sqrt(a.reduce((n,x)=>n+x*x,0)*b.reduce((n,x)=>n+x*x,0));
            if (cosine(vectors[0],vectors[1])-cosine(vectors[0],vectors[2]) < expected.embeddingSimilarityMargin) throw new Error('EMBEDDING_RELEVANCE_MISMATCH');
          }
          let wordErrorRate;
          if (expected.transcript) {
            const reference = expected.transcript.toLowerCase().match(/\\w+/g) ?? [];
            const actual = text.match(/\\w+/g) ?? [];
            let previous = actual.map((_,i)=>i); previous.push(actual.length);
            for (let i=1;i<=reference.length;i++) {
              const current=[i];
              for (let j=1;j<=actual.length;j++) current.push(Math.min(current[j-1]+1,previous[j]+1,previous[j-1]+(reference[i-1]===actual[j-1]?0:1)));
              previous=current;
            }
            wordErrorRate=previous[actual.length]/Math.max(1,reference.length);
            if (wordErrorRate>expected.maximumWordErrorRate) throw new Error('TRANSCRIPT_WORD_ERROR_RATE_FAILED');
          }
          let imageFixture;
          if (expected.imageColor) {
            const authorized = await api.getArtifactUrl(output.artifact.id);
            if (!authorized.ok) throw new Error(authorized.error.code);
            const image = new Image(); image.crossOrigin='anonymous';
            await new Promise((resolve,reject)=>{image.onload=resolve;image.onerror=()=>reject(new Error('ARTIFACT_IMAGE_UNREADABLE'));image.src=authorized.value;});
            const canvas=document.createElement('canvas');canvas.width=image.width;canvas.height=image.height;
            const context=canvas.getContext('2d');context.drawImage(image,0,0);
            const pixels=context.getImageData(0,0,image.width,image.height).data;
            const axis=expected.imageColor==='red'?0:2;let count=0,minX=image.width,minY=image.height,maxX=-1,maxY=-1;
            for(let i=0;i<pixels.length;i+=4) if(pixels[i+axis]>120 && pixels[i+axis]-Math.max(pixels[i+(axis===0?1:0)],pixels[i+(axis===0?2:1)])>60) {
              count++;const x=(i/4)%image.width,y=Math.floor(i/4/image.width);minX=Math.min(minX,x);maxX=Math.max(maxX,x);minY=Math.min(minY,y);maxY=Math.max(maxY,y);
            }
            const area=count/(image.width*image.height),aspect=(maxX-minX+1)/(maxY-minY+1),fill=count/((maxX-minX+1)*(maxY-minY+1));
            if(area<0.1||area>0.8||aspect<0.9||aspect>1.1||fill<0.95) throw new Error('REQUESTED_IMAGE_COLOR_OR_SHAPE_FAILED');
            imageFixture={color:expected.imageColor,areaFraction:area,squareFill:fill,width:image.width,height:image.height};
          }
          if (expected.topRetrievalQuery !== undefined) {
            const scores = output.value.queries.map(q => q.score);
            if (scores.indexOf(Math.max(...scores)) !== expected.topRetrievalQuery || Math.abs(scores[0]-scores[1]) <= 0.5) throw new Error('RETRIEVAL_MISMATCH');
          }
          let videoFixture;
          if (expected.videoGeometry) {
            const authorized = await api.getArtifactUrl(output.artifact.id);
            if (!authorized.ok) throw new Error(authorized.error.code);
            const video = document.createElement('video'); video.crossOrigin='anonymous'; video.preload='auto';
            await new Promise((resolve,reject)=>{
              const timer=setTimeout(()=>reject(new Error('VIDEO_DECODE_TIMEOUT')),15000);
              video.onloadeddata=()=>{clearTimeout(timer);resolve();};
              video.onerror=()=>{clearTimeout(timer);reject(new Error('VIDEO_UNREADABLE'));};
              video.src=authorized.value;
            });
            const geometry=expected.videoGeometry;
            if(video.videoWidth!==geometry.width || video.videoHeight!==geometry.height || !Number.isFinite(video.duration) || Math.abs(video.duration-geometry.frames/input.fps)>1/input.fps) throw new Error('VIDEO_GEOMETRY_MISMATCH');
            videoFixture={width:video.videoWidth,height:video.videoHeight,durationSeconds:video.duration};
            if(expected.videoSubjectColor==='red') {
              await new Promise((resolve,reject)=>{
                const timer=setTimeout(()=>reject(new Error('VIDEO_SEEK_TIMEOUT')),10000);
                video.onseeked=()=>{clearTimeout(timer);resolve();};video.currentTime=video.duration/2;
              });
              const canvas=document.createElement('canvas');canvas.width=video.videoWidth;canvas.height=video.videoHeight;
              const context=canvas.getContext('2d');context.drawImage(video,0,0);
              const pixels=context.getImageData(0,0,canvas.width,canvas.height).data;let red=0;
              for(let i=0;i<pixels.length;i+=4) if(pixels[i]>120 && pixels[i]-Math.max(pixels[i+1],pixels[i+2])>50) red++;
              videoFixture.redSubjectFraction=red/(canvas.width*canvas.height);
              if(videoFixture.redSubjectFraction<0.01) throw new Error('VIDEO_REQUESTED_SUBJECT_NOT_VISIBLE');
            }
            video.removeAttribute('src');video.load();
          }
          let cancellation;
          if (${JSON.stringify(test.cancelInWorkspace === true)} && ${repeat} === 0) {
            let stopCancel, timeoutCancel, requested = false;
            const cancelled = new Promise((resolve,reject) => {
              timeoutCancel = setTimeout(() => {stopCancel?.(); reject(new Error('WORKSPACE_CANCEL_TIMEOUT'));}, 60000);
              stopCancel = api.onJobEvent(event => {
                if (!requested && event.event === 'job.progress' && event.data?.stage === ${JSON.stringify(test.cancellationStage ?? 'running-model')}) {
                  requested = true;
                  void api.cancel(event.jobId).then(result => {if(!result.ok) reject(new Error(result.error.code));});
                }
                if (['job.completed','job.failed','job.cancelled'].includes(event.event)) {
                  clearTimeout(timeoutCancel);stopCancel();
                  if (event.event !== 'job.cancelled' || !requested) reject(new Error('WORKSPACE_CANCEL_FAILED'));
                  else resolve({terminal: event.event, requestedAtStage: ${JSON.stringify(test.cancellationStage ?? 'running-model')}});
                }
              });
            });
            const next = await api.invoke(input);
            if (!next.ok) {clearTimeout(timeoutCancel);stopCancel();throw new Error(next.error.code);}
            cancellation = await cancelled;
          }
          for (const id of owned) {
            const released = await api.releaseArtifact(id);
            const denied = await api.getArtifactUrl(id);
            if (!released.ok || denied.ok) throw new Error('ARTIFACT_RELEASE_FAILED');
          }
          if (website) {
            const released = await bridge.releaseModelSession(session.value.sessionId);
            if (!released.ok) throw new Error('WEBSITE_SESSION_RELEASE_FAILED');
            const denied = await api.invoke(input);
            if (denied.ok) throw new Error('WEBSITE_RELEASED_SESSION_REUSED');
          }
          return {output, audioMetrics, wordErrorRate, forecastMAE, tabularMAE, tabularAccuracy, imageFixture, videoFixture, textDeltaCount, cancellation, elapsedMs: performance.now()-started, isolated: true,
            runtimeFieldsRejected: true, artifactReleaseEnforced: true, state};
        })()`);
        (row.samples as unknown[]).push(sample);
        await controller.closeTab();
        tabOpen = false;
        save();
      }
      row.status = 'passed';
    } catch (error) {
      row.status = 'failed';
      row.error = String(error);
    } finally {
      if (tabOpen) await controller.closeTab();
      if (preparedModel) {
        try {
          // Closing the workspace releases its session; residency is a separate
          // lifecycle. Unload before preparing another task on this checkpoint.
          const unloaded = await engine.unloadModel(
            `unload_acceptance_${results.length}`,
            preparedModel,
            'fail-if-busy',
          );
          row.unload = unloaded;
          if (unloaded.state !== 'unloaded') {
            row.status = 'failed';
            row.cleanupError = 'ACCEPTANCE_UNLOAD_FAILED';
          }
        } catch (error) {
          row.status = 'failed';
          row.cleanupError = String(error);
        }
      }
    }
    save();
  }
  const failures = results.filter((row) => row.status !== 'passed').length;
  return { passed: failures === 0, cases: results.length, failures };
}
