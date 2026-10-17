export const meta = {
  name: 'neuron-audit-iv1r-slice',
  description: 'One slice of the blind Haiku audit of InceptionV1 units against random directions (our renderer) (rate, then held-out description test); args = [{job: "n000:A", desc: null|string}]',
  phases: [
    { title: 'Rate', detail: 'blind Haiku rater' },
    { title: 'Test', detail: 'blind Haiku judge finds the held-out crops from the description' },
  ],
}

const DIR = '/tmp/claude-0/-home-user-rcdltesting/67502866-e3ee-5513-b270-94b7a7c21d08/scratchpad/audit_iv1r/items'
const CATS = ['one_job', 'one_job_related_offshoots', 'two_related_jobs', 'several_unrelated', 'not_understood']

const RATE = {
  type: 'object',
  properties: {
    pos_description_no_nouns: { type: 'string', description: 'what the positive pole responds to, without a list of object nouns' },
    pos_description: { type: 'string', description: 'at most 40 words; a stranger must be able to recognise new crops of this neuron from it' },
    pos_category: { type: 'string', enum: CATS },
    pos_coverage: { type: 'integer', description: 'how many of the 16 crops in sections A and B the description covers (0-16)' },
    pos_confidence: { type: 'integer', description: '1 (guess) to 5 (certain)' },
    neg_description: { type: 'string' },
    neg_category: { type: 'string', enum: CATS.concat(['none_or_weak']) },
    reasoning: { type: 'string', description: 'at most 80 words' },
  },
  required: ['pos_description_no_nouns', 'pos_description', 'pos_category', 'pos_coverage', 'pos_confidence', 'neg_description', 'neg_category', 'reasoning'],
}
const TEST = {
  type: 'object',
  properties: {
    picks: { type: 'array', items: { type: 'integer' }, description: 'exactly 8 distinct crop numbers from 1 to 16' },
    confidence: { type: 'integer', description: '1 (guess) to 5 (certain)' },
  },
  required: ['picks', 'confidence'],
}

const ratePrompt = (item) => `You are auditing one neuron of an image classifier, as an interpretability researcher would.

Read the image ${DIR}/${item}_panel.png and the text file ${DIR}/${item}_cells.txt (use the Read tool for both; read
nothing else). The image shows numbered image crops on which the neuron responds, in four sections:
  A: its strongest activations;  B: around the 99.9th percentile;  C: around the 99th percentile (weak);
  D: its most negative activations.
Each crop is centred on the place where the neuron responds most. The text file gives each crop's ImageNet label and its
activation relative to the strongest crop (D's values show how strong the negative pole is).

Method:
1. Look at the crops themselves, not only at the labels. First try to say what the POSITIVE pole (A and B) responds to
   WITHOUT a list of object nouns: a shape, texture, part, pose, colour pattern, layout or other property the crops share.
   Connections can be fairly abstract (for example a neuron for a V shape fires on cat ears and on the fronts of cars).
   One pattern described as a list of nouns counts as one job. A list of nouns with no shared property does not.
2. A main job plus weaker, adjacent cases (offshoots) still counts as one main job.
3. Weak activations (C) count much less than strong ones (A, B).
4. Judge the negative pole (D) on its own.
5. Be strict and honest. Do not invent a vague umbrella ("objects", "textures", "animals", "things in scenes") that would
   fit almost any neuron. If no single property covers most of A and B, say so.
Categories:
  one_job: one property covers nearly all of A and B.
  one_job_related_offshoots: one main property, plus a minority of related, adjacent cases.
  two_related_jobs: two distinct properties that are related to each other.
  several_unrelated: two or more properties with no clear relation.
  not_understood: no hypothesis covers most of A and B.
Return the structured fields. pos_description must let a stranger pick out NEW crops of this neuron from crops of other
neurons, so make it specific.`

const testPrompt = (item, desc) => `One neuron of an image classifier responds strongly to the following (a description
written by someone who studied its top crops):

"${desc}"

Read the image ${DIR}/${item}_test.png (use the Read tool; read nothing else). It shows 16 numbered crops. Exactly 8 of
them are strong activations of this neuron; the other 8 are strong activations of other neurons of the same layer. Pick
the 8 crops that fit the description best. Return exactly 8 distinct numbers from 1 to 16.`

const jobs = (args || []).map(x => ({ item: x.job.split(':')[0], rater: x.job.split(':')[1], desc: x.desc }))
const results = await pipeline(jobs,
  j => j.desc ? Promise.resolve({ pos_description: j.desc, reused: true })
    : agent(ratePrompt(j.item), { label: `rate:${j.item}:${j.rater}`, phase: 'Rate', schema: RATE, model: 'haiku' }),
  (r, j) => r ? agent(testPrompt(j.item, r.pos_description), { label: `test:${j.item}:${j.rater}`, phase: 'Test', schema: TEST, model: 'haiku' })
    .then(t => (t ? 1 : 0)) : 0,
)
return { jobs: jobs.length, tested: results.filter(x => x === 1).length }
