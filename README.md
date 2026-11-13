# SafetyDrift / DriftWatch demo

Live site: https://fhisheruser.github.io/safetydrift/

An interactive demo of DriftWatch, a monitor that tells you *which part* of a deployed language model made it less safe: the model, the chat context, the user population, or the safety judge itself.

- Pick something to break, press Start, and watch four evidence meters build toward an alarm.
- The verdict names the cause and the fix.
- "Does it cry wolf?" runs 100 drift-free streams to show the false-alarm rate.

The two safety judges behind the simulation were trained on PKU-SafeRLHF; their per-topic hit and miss rates drive every simulated reply. The drift itself is injected, which is how the right answer is known. Everything runs in the browser; there is no server.
