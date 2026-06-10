# ns-3 / ns3-ai provenance (git metadata removed for a single-repo workspace)

- ns-3-dev: official ns-3.42 release, upstream https://gitlab.com/nsnam/ns-3-dev.git,
  commit ab4cce021 ("Update VERSION and documentation tags for ns-3.42 release"),
  pristine (no local modifications).
- ns3-ai (ns-3-dev/contrib/ai): upstream https://github.com/hust-diangroup/ns3-ai.git,
  commit b8c9858. Local change: examples/CMakeLists.txt (disabled rate-control & multi-bss
  examples — API drift with ns-3.42). Saved as ns3ai_local_changes.patch.

To restore git tracking later: re-clone at these commits and re-apply the patch.
