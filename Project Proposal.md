# Part I

1. ## Problem and Motivation

   Students often enter math courses carrying wrong beliefs that are hard to dislodge. A student who writes 1/2 \+ 1/3 \= 2/5 has internalized a procedure that feels consistent. Pointing to the correct answer rarely helps; what works is getting the student to see why their own reasoning breaks down, then verifying the fix carried over to a new problem. Macina et al. (2023) show that LLMs fail at this specifically because they either give wrong feedback or hand over the answer, cutting the reasoning short \[1\]. We want to build an agent that works through the misconception with the student rather than around it.

2. ## Our Approach

   Given a student with a known misconception, the agent runs a multi-turn dialogue to correct it. It asks the student to explain their reasoning, surfaces the flaw, then presents a counter-example or guided question to create conflict with the wrong belief. Finally it gives a transfer problem to check whether the correction stuck. The key design choice is keeping diagnosis and correction separate: our agent tracks what it has addressed and adapts each turn accordingly.

3. ## Evaluation Overview

   We evaluate on a 20-task benchmark using an LLM-based student simulator. Success is whether the student answers a transfer question correctly after the session, checked automatically. We compare against a direct-explanation baseline and a chain-of-thought tutor with no state tracking.

# Part II: Benchmark Design

1. ## Task Domain

   The benchmark covers K-12 math across four areas: fractions, negative numbers, algebra, and geometry. Each has well-studied misconceptions with consistent error patterns, making it straightforward to define simulator behavior and check answers automatically.

2. ## Task Specifications

   20 tasks total, 5 per topic, ranging in difficulty based on how resistant the misconception is to correction (calibrated through pilot testing). Each task includes the misconception, simulator behavior, a transfer question, and the correct answer. Three examples:  
     
   Example 1 (Fractions): Misconception: fraction addition adds numerators and denominators separately (1/2 \+ 1/3 \= 2/5). Transfer test: 1/3 \+ 1/4. Correct answer: 7/12.  
     
   Example 2 (Negative Numbers): Misconception: multiplying two negatives gives a negative. Transfer test: (negative 4\) times (negative 3). Correct answer: 12\.  
     
   Example 3(Algebra): Misconception: exponents distribute over addition, so (a \+ b)^2 \= a^2 \+ b^2. Transfer test: expand (x \+ 3)^2. Correct answer: x^2 \+ 6x \+ 9\. 

   These tasks are challenging because the agent must infer the misconception from sparse evidence before correcting it, without revealing the answer. Harder tasks involve misconceptions that feel analogous to valid rules and resist straightforward counter-examples.

3. ## Success Criteria

   Primary metric: does the simulator answer the transfer question correctly after the session, checked automatically against the stored answer. Secondary metrics are turns to correction and whether the agent diagnosed before correcting.

4. ## Benchmark Independence

   All tasks are stored as JSON with the schema {id, topic, misconception, simulator prompt, transfer question, correct answer, max turns}. The simulator is a standalone module; any agent can be evaluated by swapping in their own implementation against the same tasks. The only dependency is the Gemini API. We will release all task files, the simulator code, and a README with instructions for running the evaluation on GitHub.

# Part III: Agent Design

1. ## Agent Architecture

   Four components: Misconception Tracker (JSON state of what has been addressed), Dialogue Planner (picks next strategy from tracker state), Response Generator (writes the tutor turn), and Turn Manager (enforces turn limit, triggers transfer test).

2. ## Technical Approach

   After each student response, Gemini 3 Pro updates the tracker to note whether the misconception has been identified, a counter-example shown, and whether the student has started to shift. The planner picks one of three moves: probing (ask the student to walk through their reasoning), confronting (show where the wrong rule breaks down), or confirming (try a simpler variant to check partial progress). The tutor turn is generated using the dialogue history and chosen move. The simulator runs on Gemini 3 Flash with a fixed system prompt. If the tracker returns malformed JSON, the agent falls back to probing before retrying.

3. ## Baseline Comparisons

   Both baselines use Gemini 3 Pro:  
- Baseline 1: Direct explanation. One response, no follow-up.  
- Baseline 2: Chain-of-thought tutor. Prompted to think before responding, no state tracking.  
  These baselines represent how LLMs are most commonly deployed as tutors. Baseline 1 isolates the value of multi-turn interaction; Baseline 2 isolates the value of explicit state tracking over unstructured reasoning.

# Part IV: Resources and Timeline

1. ## Resources

- Google Gemini API (Vertex AI): Gemini 3 Pro Preview for the tutor agent and baselines, Gemini 3 Flash for the student simulator. Estimated cost: free, covered by available Google Cloud credits ($300).  
- Python for orchestration and evaluation. No GPU or external datasets required.

2. ## Timeline

     
   Feb 28: 20 tasks finalized, simulator prompt written.  
   Mar 5  Project Update 1: Benchmark tasks drafted, simulator prototype running.  
   Mar 12 Evaluation Plan: Scoring script and simulator tested end-to-end.  
   Mar 14: Both baselines implemented.  
   Mar 17–21: Agent core built: tracker, planner, and generator.  
   Mar 26 Project Update 2: Full benchmark run across all three systems. Results collected.  
   Apr 7  Draft Benchmark Paper: Benchmark design and results submitted.  
   Apr 9  Project Update 3: Agent implementation complete, analysis underway.  
   Apr 16  Draft Agent Paper: Agent design, baselines, and analysis submitted.  
   Apr 28  May 5 \- Final Presentations.  
   May 5 Final Benchmark and Agent Papers due.  
   

# References

1.  Macina, Jakub et al. "MathDial: A Dialogue Tutoring Dataset with Rich Pedagogical Properties Grounded in Math Reasoning Problems." Findings of EMNLP 2023\.  
2. Macina, Jakub et al. "Opportunities and Challenges in Neural Dialog Tutoring." EACL 2023\.

