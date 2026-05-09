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

## 发布检查清单

1. 确认工作区干净。
2. 更新 `pyproject.toml` 中的 `version`。
3. 将 `CHANGELOG.md` 中 `未发布` 的内容移动到本次发布版本。
4. 运行 `uv run pytest`。
5. 运行 `uv build`。
6. 创建并发布 GitHub Release，tag 使用 `vX.Y.Z`。
7. 等待 `Publish to PyPI` workflow 完成。
8. 从 PyPI 重新安装并验证：

```bash
uv tool install meeting-asr --python 3.14 --reinstall --refresh
meeting-asr --version
```

## 为什么使用 Trusted Publishing

Trusted Publishing 让 GitHub Actions 通过 OIDC 获取短期 PyPI 发布凭证。
这样 workflow 不需要长期 PyPI Token，即使仓库 secrets 泄露，也不会直接泄露 PyPI 发布凭证。
