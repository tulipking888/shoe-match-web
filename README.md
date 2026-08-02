# AI 样品识别系统

纯前端、可离线使用的样品图片检索工具，支持 Excel 内嵌图片批量导入、手机拍照或图片文件查询，并返回 Top 5 相似样品。

## V4 本地 AI 视觉检索

- 使用量化 DINOv2-small 模型生成 384 维视觉向量，替代感知哈希、全局颜色和边缘统计。
- 模型通过 ONNX Runtime Web 在浏览器本机运行，照片不会上传服务器。
- 模型与运行库由 Service Worker 缓存；完成首次加载和样品导入后可离线查询。
- 同一编号的多张样品图全部保留，查询时按编号合并并采用最高匹配得分。
- V3 及更早版本的图片特征不兼容，需要清空旧数据后重新导入 Excel。
- 384 维向量和缩略图保存在当前浏览器 IndexedDB 中。

## V3 企业内部数据管理

- 样品编号允许重复，每条记录由数据库内部 ID 区分。
- 删除了“存放位置”字段。
- 每次 Excel 导入自动生成独立批次。
- 批次记录包含文件名、文件大小、导入时间、成功数、失败数和失败行。
- 可删除指定导入批次后重新导入，其他批次不受影响。
- 批次删除后仍保留“已删除”审计记录。
- 检测相同文件重复导入并进行提醒。
- 支持下载导入失败报告。
- 清空全部样品必须输入 `DELETE` 二次确认。
- 支持显示浏览器存储使用量。
- 支持完整 JSON 备份和恢复。

## Excel 字段

支持以下列名：

- 样品编号
- 样品寄出时间
- 客户编号
- 订单编号
- 特别要求
- 图片（Excel 内嵌图片）

## 部署

将以下文件放到 GitHub Pages 或 Cloudflare Pages 的站点根目录：

```text
index.html
app.js
styles.css
jszip.min.js
vendor/onnxruntime/
models/dinov2-small/
manifest.webmanifest
sw.js
```

无需业务服务器。样品及图片特征保存在当前浏览器 IndexedDB 中，因此不同设备不会自动同步；更换设备前请导出完整备份。

## 第三方组件

- DINOv2-small：Meta AI，Apache-2.0。仓库内模型为 ONNX INT8 量化版本。
- ONNX Runtime Web：Microsoft，MIT。
