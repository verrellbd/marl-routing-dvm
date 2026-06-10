# Cluster Setup Notes (malmo)

## Working State (May 21, 2026)
- ns-3.42, ns3-ai commit b8c9858
- Python 3.9 venv at ~/thesis/ns3ai-venv
- Protobuf from Anaconda (/opt/anaconda3)
- Disabled examples (API drift with ns-3.42): rate-control, multi-bss

## Key environment variables (in .bashrc)
- PATH includes /usr/local/cuda/bin
- LD_LIBRARY_PATH includes /usr/local/cuda/lib64
- PYTHONPATH includes ~/thesis/ns-3-dev/contrib/ai/model/gym-interface/py
- Venv auto-activates

## Sanity check
cd ~/thesis/ns-3-dev/contrib/ai/examples/a-plus-b/use-gym
python apb.py
# Expected: "set: X,Y; get: Z;" repeating, then "Experiment destroyed"

## Rebuild from scratch
1. Restore Python venv: pip install -r ~/thesis/requirements.txt
2. Reinstall ns3-ai editable: pip install -e ~/thesis/ns-3-dev/contrib/ai/python_utils/
3. Reinstall gym env: pip install -e ~/thesis/ns-3-dev/contrib/ai/model/gym-interface/py/
4. Configure: ~/thesis/configure_ns3.sh
5. Build: cd ~/thesis/ns-3-dev && ./ns3 build -j8
