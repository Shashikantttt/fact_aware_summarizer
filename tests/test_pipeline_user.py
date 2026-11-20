import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline import ExplainableSummarizationPipeline

text = """The external affairs minister also flagged terrorism as another major threat to global peace, saying its use as an "instrument of statecraft" was "simply unacceptable" and could neither be justified nor rationalised.

"Terrorism is another deadly threat to peace," he said, adding that the threat continued because not all countries displayed zero tolerance towards it.

"Despite history holding so many lessons, some still allow political calculations to take precedence," Jaishankar said, calling for greater global awareness and stronger collective action to counter terrorism in all its forms.

Linking global peace to economic development, Jaishankar said progress and prosperity were equally interconnected, particularly in a world dependent on global supply chains, markets and shared resources.

“Progress and prosperity are the flip side of conflict and violence,” the minister said.

"The world cannot advance half prosperous and half denied in an era of intricate supply chains, of globalized markets, and of common resources. This is simply a prescription for continuous instability," he added."""

pipeline = ExplainableSummarizationPipeline(mock_mode=False)
res = pipeline.process(text, extractive_sentences=2)

print("\n=== INPUT METRICS ===")
print(f"Total Sentences: {res['source']['total_sentences']}")
print(f"Total Words: {len(text.split())}")

print("\n=== 1. EXTRACTIVE SUMMARY (TextRank) ===")
print(res['extractive']['textrank']['summary'])
print(f"Words: {len(res['extractive']['textrank']['summary'].split())} ({int((1 - len(res['extractive']['textrank']['summary'].split())/len(text.split()))*100)}% compression)")

print("\n=== 2. ABSTRACTIVE SUMMARY ===")
print(res['abstractive']['summary'])
print(f"Words: {len(res['abstractive']['summary'].split())} ({int((1 - len(res['abstractive']['summary'].split())/len(text.split()))*100)}% compression)")

print("\n=== 3. VERIFICATION & GROUNDING ===")
for i, c in enumerate(res['verification']['claims']):
    print(f"Claim {i+1} [{c['badge'].upper()}]: {c['claim']}")
    for ev in c.get('evidence_sources', []):
        print(f"   -> Grounding Source #{ev['source_index']+1}: {ev['sentence'][:70]}... (Sim: {ev['similarity_score']})")
