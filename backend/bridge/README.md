# FIRE Agent frontend bridge

This service connects the Sites/Next frontend to the real FIRE Agent Harness in
`src/fire_agent`. It does not contain a second research implementation.

Request modes:

- `deep`: the normal multi-round FIRE Agent with research tools and a public,
  source-aware timeline.
- `fast`: the Harness-native `final_answer_only_round` with `max_steps=1` and
  no external tools. The frontend receives the model's native thinking content
  for a collapsible preview, followed by the final answer stream.

The 27B model is presented publicly as **Mint-Ag**. Its internal model id and
vLLM serving name remain `mint-sg` / `Mint-Sg` for runtime compatibility.

Start it on the harness machine:

```bash
/home/test/miniconda3/envs/gyz/bin/python \
  /home/test/gyz/mint-agent/bridge/fire_agent_bridge.py \
  --host 0.0.0.0 --port 4180
```

The bridge reads the project `.env`. Set a dedicated
`FIRE_AGENT_BRIDGE_TOKEN` for production; if absent it falls back to
`FIRE_AGENT_API_KEY` so existing local development continues to work.

Frontend runtime variables:

```text
FIRE_AGENT_BRIDGE_URL=https://your-bridge.example
FIRE_AGENT_BRIDGE_TOKEN=...
```

Health check:

```bash
curl http://127.0.0.1:4180/health
```
