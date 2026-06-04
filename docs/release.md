# 发布流程

Meeting-ASR 通过 GitHub Release 触发 PyPI Trusted Publishing 发布到 PyPI。
不要在 GitHub Secrets 中保存 PyPI API Token。

## 一次性 PyPI 配置

创建或认领 PyPI 项目 `meeting-asr`，然后添加 GitHub Actions trusted publisher。
如果项目还不存在，可以先创建 pending trusted publisher，配置如下：

- Owner: `crhan`
- Repository: `meeting-asr`
- Workflow file: `publish.yml`
- Environment: `pypi`

GitHub environment `pypi` 是有意保留的。请在 GitHub 仓库设置中配置 `pypi` environment；如果仓库是多人协作，建议给该 environment 配置发布审批人。

## 发布检查清单（直接 main + tag，不走 PR）

发版只是把已经评审合入 `main` 的内容固化一个版本号，再走一遍 PR 是多余仪式。
版本固化 commit **直接推 `main`**——这是 `git push origin main` 的唯一豁免（见 `AGENTS.md`
的「发版豁免」）；功能开发仍走「分支 + PR」。

1. 确认在主 checkout 的 `main`，`git pull --ff-only origin main` 已最新，工作区干净。
2. 更新 `pyproject.toml` 的 `version`，并 `uv lock` 同步 `uv.lock` 里的 `meeting-asr` 版本字段。
3. 把 `CHANGELOG.md` 的变更固化为 `[X.Y.Z] - YYYY-MM-DD` 小节。
4. `uv run pytest` 与 `uv build`，两者都要绿。
5. 直接 commit 并 `git push origin main`。
6. 打 tag 并创建 GitHub Release（创建即触发发布）：

```bash
gh release create vX.Y.Z --title vX.Y.Z --notes-file /private/tmp/meeting-asr-X.Y.Z-notes.md
```

   - 发布说明必须使用中文，从 `CHANGELOG.md` 对应版本小节提取，不要临时手写英文摘要。
   - notes 文件写到 `/private/tmp/meeting-asr-X.Y.Z-notes.md`，内容仍必须是中文。

7. 等待 `Publish to PyPI` workflow 完成。
8. 从 PyPI 重新安装并验证：

```bash
uv tool install meeting-asr --python 3.14 --reinstall --refresh
meeting-asr --version
```

## 为什么使用 Trusted Publishing

Trusted Publishing 让 GitHub Actions 通过 OIDC 获取短期 PyPI 发布凭证。
这样 workflow 不需要长期 PyPI Token，即使仓库 secrets 泄露，也不会直接泄露 PyPI 发布凭证。
