"""Can't-miss ("red flag") diagnoses, compiled from standard emergency-medicine references.

Each entry: canonical name -> aliases used for string matching against a gold diagnosis, plus the
sentinel presenting feature a competent clinician must react to. Matching is a *candidate filter*
only; every candidate is hand-verified before entering RedFlag-N (see scripts/build_redflag.py).
"""

RED_FLAGS: dict[str, dict] = {
    "subarachnoid hemorrhage": dict(
        aliases=["subarachnoid hemorrhage", "subarachnoid haemorrhage", "sah", "ruptured aneurysm",
                 "berry aneurysm rupture"],
        sentinel="sudden thunderclap / 'worst headache of my life', maximal at onset"),
    "aortic dissection": dict(
        aliases=["aortic dissection", "dissecting aortic aneurysm", "type a dissection",
                 "type b dissection"],
        sentinel="tearing chest/back pain, pulse or blood-pressure differential"),
    "ruptured abdominal aortic aneurysm": dict(
        aliases=["abdominal aortic aneurysm", "ruptured aaa", "aaa rupture"],
        sentinel="older smoker, pulsatile mass, hypotension with back/flank pain"),
    "pulmonary embolism": dict(
        aliases=["pulmonary embolism", "pulmonary embolus", "pe", "saddle embolus"],
        sentinel="pleuritic pain/dyspnoea with hypoxia, tachycardia, immobilisation or malignancy"),
    "acute coronary syndrome": dict(
        aliases=["myocardial infarction", "acute coronary syndrome", "stemi", "nstemi",
                 "unstable angina", "acute mi"],
        sentinel="exertional or ongoing chest pressure with diaphoresis or ECG change"),
    "cardiac tamponade": dict(
        aliases=["cardiac tamponade", "pericardial tamponade"],
        sentinel="hypotension, JVD, muffled heart sounds, pulsus paradoxus"),
    "tension pneumothorax": dict(
        aliases=["tension pneumothorax"],
        sentinel="unilateral absent breath sounds with hypotension and tracheal deviation"),
    "sepsis / septic shock": dict(
        aliases=["sepsis", "septic shock", "severe sepsis", "bacteremia with shock"],
        sentinel="infection source with hypotension, tachycardia, altered mentation or lactate rise"),
    "bacterial meningitis": dict(
        aliases=["bacterial meningitis", "meningococcal meningitis", "meningitis",
                 "meningococcemia", "meningococcal septicemia"],
        sentinel="fever with neck stiffness, altered mental status, petechial rash"),
    "encephalitis": dict(
        aliases=["herpes simplex encephalitis", "hsv encephalitis", "encephalitis"],
        sentinel="fever with focal neurologic deficit or seizure and altered mentation"),
    "ectopic pregnancy": dict(
        aliases=["ectopic pregnancy", "tubal pregnancy", "ruptured ectopic"],
        sentinel="reproductive-age abdominal pain with amenorrhoea; pregnancy test mandatory"),
    "ovarian torsion": dict(
        aliases=["ovarian torsion", "adnexal torsion"],
        sentinel="sudden unilateral pelvic pain with adnexal mass"),
    "testicular torsion": dict(
        aliases=["testicular torsion", "torsion of the testis", "spermatic cord torsion"],
        sentinel="acute testicular pain, absent cremasteric reflex, high-riding testis"),
    "necrotizing fasciitis": dict(
        aliases=["necrotizing fasciitis", "necrotising fasciitis", "fournier gangrene",
                 "gas gangrene", "necrotizing soft tissue infection"],
        sentinel="pain out of proportion, crepitus, rapidly advancing erythema, systemic toxicity"),
    "epiglottitis": dict(
        aliases=["epiglottitis", "acute epiglottitis", "supraglottitis"],
        sentinel="drooling, tripod posture, stridor, muffled voice"),
    "cauda equina syndrome": dict(
        aliases=["cauda equina syndrome", "cauda equina compression"],
        sentinel="saddle anaesthesia, urinary retention, bilateral leg symptoms"),
    "spinal epidural abscess": dict(
        aliases=["spinal epidural abscess", "epidural abscess", "vertebral osteomyelitis"],
        sentinel="back pain with fever and neurologic deficit; IVDU or bacteraemia"),
    "cord compression": dict(
        aliases=["spinal cord compression", "malignant cord compression"],
        sentinel="known malignancy with progressive back pain and neurologic level"),
    "giant cell arteritis": dict(
        aliases=["giant cell arteritis", "temporal arteritis", "gca"],
        sentinel="new headache >50y with jaw claudication or visual loss; ESR/CRP"),
    "acute angle-closure glaucoma": dict(
        aliases=["acute angle closure glaucoma", "angle-closure glaucoma", "acute glaucoma"],
        sentinel="painful red eye, mid-dilated fixed pupil, haloes, vomiting"),
    "central retinal artery occlusion": dict(
        aliases=["central retinal artery occlusion", "crao", "retinal artery occlusion"],
        sentinel="sudden painless monocular vision loss"),
    "retinal detachment": dict(
        aliases=["retinal detachment"],
        sentinel="flashes, floaters, curtain over the visual field"),
    "acute stroke": dict(
        aliases=["ischemic stroke", "ischaemic stroke", "cerebrovascular accident", "acute stroke",
                 "intracerebral hemorrhage", "intracranial hemorrhage", "basilar occlusion"],
        sentinel="sudden focal deficit; time-critical thrombolysis window"),
    "diabetic ketoacidosis": dict(
        aliases=["diabetic ketoacidosis", "dka"],
        sentinel="hyperglycaemia with anion-gap acidosis and ketones"),
    "hyperosmolar hyperglycemic state": dict(
        aliases=["hyperosmolar hyperglycemic state", "hhs", "hyperosmolar nonketotic"],
        sentinel="marked hyperglycaemia with hyperosmolality and altered mentation"),
    "adrenal crisis": dict(
        aliases=["adrenal crisis", "addisonian crisis", "acute adrenal insufficiency"],
        sentinel="shock refractory to fluids with hyponatraemia and hyperkalaemia"),
    "thyroid storm": dict(
        aliases=["thyroid storm", "thyrotoxic crisis"],
        sentinel="fever, tachyarrhythmia, agitation on a thyrotoxic background"),
    "myxedema coma": dict(
        aliases=["myxedema coma", "myxoedema coma"],
        sentinel="hypothermia, bradycardia, hyponatraemia, depressed mentation"),
    "acute cholangitis": dict(
        aliases=["ascending cholangitis", "acute cholangitis", "cholangitis"],
        sentinel="Charcot triad: fever, jaundice, RUQ pain"),
    "mesenteric ischemia": dict(
        aliases=["mesenteric ischemia", "mesenteric ischaemia", "acute mesenteric ischemia"],
        sentinel="pain out of proportion to examination, AF or vascular disease, lactate rise"),
    "perforated viscus": dict(
        aliases=["perforated peptic ulcer", "bowel perforation", "perforated viscus",
                 "gastrointestinal perforation"],
        sentinel="sudden severe pain with peritonism and free air"),
    "volvulus": dict(
        aliases=["sigmoid volvulus", "cecal volvulus", "midgut volvulus", "volvulus"],
        sentinel="obstruction with rapid distension; bilious vomiting in an infant"),
    "intussusception": dict(
        aliases=["intussusception"],
        sentinel="intermittent inconsolable crying, currant-jelly stool, sausage-shaped mass"),
    "compartment syndrome": dict(
        aliases=["compartment syndrome", "acute compartment syndrome"],
        sentinel="pain out of proportion, pain on passive stretch, tense compartment"),
    "anaphylaxis": dict(
        aliases=["anaphylaxis", "anaphylactic shock", "anaphylactic reaction"],
        sentinel="rapid multi-system allergic reaction with airway or circulatory compromise"),
    "status epilepticus": dict(
        aliases=["status epilepticus"],
        sentinel="seizure >5 min or repeated seizures without recovery"),
    "neuroleptic malignant syndrome": dict(
        aliases=["neuroleptic malignant syndrome", "nms"],
        sentinel="rigidity, hyperthermia, autonomic instability on antipsychotics"),
    "serotonin syndrome": dict(
        aliases=["serotonin syndrome", "serotonin toxicity"],
        sentinel="clonus, hyperreflexia, agitation after a serotonergic agent"),
    "malignant hyperthermia": dict(
        aliases=["malignant hyperthermia"],
        sentinel="hypercarbia, rigidity, hyperthermia after volatile anaesthetic or succinylcholine"),
    "carbon monoxide poisoning": dict(
        aliases=["carbon monoxide poisoning", "co poisoning", "carboxyhemoglobinemia"],
        sentinel="headache/confusion in multiple household members; normal SpO2"),
    "toxic shock syndrome": dict(
        aliases=["toxic shock syndrome", "tss"],
        sentinel="fever, hypotension, diffuse macular rash, desquamation"),
    "preeclampsia / HELLP / eclampsia": dict(
        aliases=["preeclampsia", "pre-eclampsia", "eclampsia", "hellp syndrome"],
        sentinel="hypertension with proteinuria, headache or RUQ pain in pregnancy"),
    "placental abruption": dict(
        aliases=["placental abruption", "abruptio placentae"],
        sentinel="painful third-trimester bleeding with a tense uterus"),
    "hyperkalemia": dict(
        aliases=["hyperkalemia", "hyperkalaemia", "severe hyperkalemia"],
        sentinel="peaked T waves or widened QRS with renal failure or potassium-sparing drugs"),
    "acute limb ischemia": dict(
        aliases=["acute limb ischemia", "acute arterial occlusion", "arterial embolism of the limb"],
        sentinel="the six Ps: pain, pallor, pulselessness, paraesthesia, paralysis, poikilothermia"),
    "esophageal rupture": dict(
        aliases=["boerhaave syndrome", "esophageal rupture", "oesophageal perforation"],
        sentinel="vomiting then severe chest pain with subcutaneous emphysema"),
    "acute liver failure": dict(
        aliases=["acute liver failure", "fulminant hepatic failure",
                 "acetaminophen toxicity", "paracetamol overdose"],
        sentinel="coagulopathy with encephalopathy; time-critical NAC"),
    "tricyclic / sodium-channel-blocker overdose": dict(
        aliases=["tricyclic antidepressant overdose", "tca overdose", "sodium channel blocker toxicity"],
        sentinel="wide QRS with altered mentation after overdose"),
    "salicylate toxicity": dict(
        aliases=["salicylate toxicity", "aspirin overdose", "salicylate poisoning"],
        sentinel="tinnitus with mixed respiratory alkalosis and metabolic acidosis"),
}


import re as _re
from functools import lru_cache

# Short acronyms match far too much as substrings ("pe" inside "peptic", "type", "pericarditis").
# They are matched only as whole words, and the very shortest are dropped entirely.
_DROP = {"pe", "mi", "co", "tss", "sah", "hhs", "gca", "nms", "dka", "crao", "aaa", "tca"}


@lru_cache(maxsize=1)
def alias_index() -> tuple[tuple[_re.Pattern, str], ...]:
    """(compiled word-boundary pattern, canonical), longest alias first so specific aliases win."""
    pairs = [(a.lower(), k) for k, v in RED_FLAGS.items() for a in v["aliases"]
             if a.lower() not in _DROP]
    pairs.sort(key=lambda p: -len(p[0]))
    return tuple((_re.compile(r"\b" + _re.escape(a) + r"\b"), k) for a, k in pairs)


def match(text: str) -> str | None:
    t = (text or "").lower()
    for pat, canon in alias_index():
        if pat.search(t):
            return canon
    return None

# ---------------------------------------------------------------------------
# Added 2026-08-28 after the RedFlag-probe confirmation pass. The hand-compiled list above missed
# these; the annotator surfaced them as genuine can't-miss conditions in real stems, so they are
# folded into the closed vocabulary rather than discarded as "unmapped".
RED_FLAGS.update({
    "acute hemolytic transfusion reaction": dict(
        aliases=["acute hemolytic transfusion reaction", "hemolytic transfusion reaction",
                 "ABO incompatibility reaction"],
        sentinel="fever, flank pain, dark urine minutes into a transfusion"),
    "septic arthritis": dict(
        aliases=["septic arthritis", "bacterial arthritis", "pyogenic arthritis"],
        sentinel="acutely hot swollen joint with fever; joint destruction within days"),
    "brain abscess": dict(
        aliases=["brain abscess", "cerebral abscess", "intracranial abscess"],
        sentinel="fever with focal deficit and ring-enhancing lesion"),
    "complete heart block": dict(
        aliases=["complete heart block", "third degree heart block", "symptomatic bradycardia",
                 "av dissociation"],
        sentinel="syncope with bradycardia and AV dissociation"),
    "brugada syndrome": dict(
        aliases=["brugada syndrome", "brugada pattern"],
        sentinel="syncope with coved ST elevation in V1-V2; sudden-death risk"),
    "organophosphate poisoning": dict(
        aliases=["organophosphate poisoning", "organophosphate toxicity",
                 "cholinesterase inhibitor poisoning", "carbamate poisoning"],
        sentinel="DUMBELS cholinergic crisis after pesticide exposure"),
    "opioid overdose": dict(
        aliases=["opioid overdose", "heroin overdose", "opiate overdose", "opioid toxicity"],
        sentinel="pinpoint pupils with respiratory depression and depressed consciousness"),
    "methemoglobinemia": dict(
        aliases=["methemoglobinemia", "methaemoglobinaemia"],
        sentinel="cyanosis with normal PaO2 and chocolate-brown blood; saturation gap"),
    "severe symptomatic hyponatremia": dict(
        aliases=["severe hyponatremia", "symptomatic hyponatremia", "beer potomania"],
        sentinel="seizure or obtundation with profound hyponatraemia"),
    "strangulated hernia": dict(
        aliases=["strangulated hernia", "incarcerated hernia", "strangulated inguinal hernia"],
        sentinel="irreducible tender hernia with obstruction or peritonism"),
    "intra-abdominal hemorrhage": dict(
        aliases=["intra-abdominal hemorrhage", "hemoperitoneum", "splenic injury",
                 "splenic rupture", "splenic laceration"],
        sentinel="blunt trauma with hypotension and free fluid"),
    "raised intracranial pressure": dict(
        aliases=["raised intracranial pressure", "increased intracranial pressure",
                 "intracranial space-occupying lesion", "brain tumor with mass effect",
                 "pituitary apoplexy"],
        sentinel="morning headache with vomiting, papilloedema or a focal deficit"),
    "superior vena cava syndrome": dict(
        aliases=["superior vena cava syndrome", "svc syndrome", "svc obstruction"],
        sentinel="facial and upper-limb swelling with distended neck veins in a smoker"),
    "status asthmaticus": dict(
        aliases=["status asthmaticus", "impending respiratory failure",
                 "near-fatal asthma", "life-threatening asthma"],
        sentinel="silent chest, exhaustion, or a rising PaCO2 in acute asthma"),
    "high-altitude cerebral edema": dict(
        aliases=["high-altitude cerebral edema", "hace", "high altitude cerebral oedema"],
        sentinel="ataxia and altered mentation after rapid ascent"),
    "diffuse alveolar hemorrhage": dict(
        aliases=["diffuse alveolar hemorrhage", "goodpasture", "anti-gbm disease",
                 "pulmonary-renal syndrome"],
        sentinel="haemoptysis with falling haemoglobin and renal impairment"),
    "hyperammonemic encephalopathy": dict(
        aliases=["hyperammonemic encephalopathy", "hyperammonemia", "urea cycle defect",
                 "ornithine transcarbamylase deficiency"],
        sentinel="encephalopathy with respiratory alkalosis and a normal glucose"),
    "gallstone ileus": dict(
        aliases=["gallstone ileus"],
        sentinel="small-bowel obstruction with pneumobilia"),
    "neuroborreliosis": dict(
        aliases=["neuroborreliosis", "lyme neuroborreliosis", "lyme meningitis",
                 "lyme carditis"],
        sentinel="facial palsy, meningitis or heart block after a tick-exposure rash"),
})

# The very short new acronyms must not be substring-matched either.
_DROP |= {"hace", "svc"}
