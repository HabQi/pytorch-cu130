# VS Code Markdown 增强预览：PDF 导出配置与修复记录

- 整理日期：2026-10-15。
- 已核对环境：Windows，当前用户 `Administrator`。
- 扩展：Markdown Preview Enhanced **0.8.39**，ID 为 `shd101wyy.markdown-preview-enhanced`。
- 导出入口：增强预览中右键 → **导出 → Chrome (Puppeteer) → PDF**。
- 本机实际调用 Microsoft Edge 的 Chromium 内核；菜单名称仍为 Chrome。
- 本文记录最终生效的方案，供迁移、插件更新后重新配置和排查问题使用，不依赖原来的 `tmp` 文件夹。

## 1. 最终导出效果

| 项目 | 最终行为 |
| --- | --- |
| 正文字体 | 英文 Times New Roman，中文宋体（SimSun） |
| 亮色主题 | PDF 文本为纯黑 `#000000` |
| 暗色主题 | PDF 文本为纯白 `#ffffff`，背景采用对应暗色主题 |
| 暗色 PDF 纸张边缘 | 与正文区域同色，不再出现外围白框 |
| 正文边距 | 普通文档每页四边默认保留 1 厘米 |
| 图片 | 正常读取包含中文的本地路径；图片原有颜色保留 |
| 数学公式 | 使用 MathJax；需完成渲染后再导出 |

这里的“暗色”指 **Markdown Preview Enhanced 实际使用的预览主题**，不是仅把 VS Code 界面改成深色。当前用户设置为 `github-light.css`；暗色效果已用 `github-dark.css` 另行测试。

## 2. 生效范围与文件位置

这些修改对当前 Windows 用户生效，切换到其他文件夹仍然有效。项目级配置可能覆盖部分用户设置。

| 内容 | 本机位置 |
| --- | --- |
| VS Code 用户设置 | `C:\Users\Administrator\AppData\Roaming\Code\User\settings.json` |
| 全局预览及打印样式 | `C:\Users\Administrator\.crossnote\style.less` |
| 已安装扩展目录 | `C:\Users\Administrator\.vscode\extensions\shd101wyy.markdown-preview-enhanced-0.8.39` |
| 本地图片路径转换修复 | 扩展目录中的 `out\native\extension.js` |
| 主题颜色及页面背景修复 | 扩展目录中的 `crossnote\styles\preview_theme\*.css` |

迁移时可用 `%APPDATA%\Code\User\settings.json` 和 `%USERPROFILE%\.crossnote\style.less` 定位；扩展目录中的版本号应以目标电脑为准。

## 3. VS Code 用户设置

通过命令面板打开 **Preferences: Open User Settings (JSON)**，合并下面这些项，不要覆盖其他设置或重复添加同名键。

下面采用标准 JSON 转义写法：文件路径中的一个反斜杠写为 `\\`，解析后是正常的 Windows 路径。

```json
{
  "markdown-preview-enhanced.chromePath": "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
  "markdown-preview-enhanced.puppeteerWaitForTimeout": 3000,
  "markdown-preview-enhanced.printBackground": true,
  "markdown-preview-enhanced.mathRenderingOption": "MathJax",
  "markdown-preview-enhanced.previewTheme": "github-light.css",
  "markdown-preview-enhanced.previewColorScheme": "selectedPreviewTheme",
  "markdown-preview-enhanced.codeBlockTheme": "auto.css"
}
```

- `chromePath`：按目标电脑的 Edge 或 Chrome 实际路径填写。本机的路径字符串当前含有重复的路径分隔符，已完成的导出可以执行；迁移时建议使用上面给出的标准写法。
- `puppeteerWaitForTimeout`：页面加载后额外等待 3000 毫秒。固定等待不能保证慢速网络下远程公式资源一定加载完成。
- `printBackground`：必须为 `true`，否则暗色背景可能不输出，导致白色文字难以辨认。
- `previewTheme`：亮色选 `github-light.css`，暗色选 `github-dark.css`。
- `previewColorScheme: selectedPreviewTheme`：以所选预览主题为准。若改成 `editorColorScheme`，还需核对插件实际导出时采用的主题。
- `codeBlockTheme: auto.css`：代码块背景随主题匹配；文字则由下文的打印规则统一为纯黑或纯白。

本机 MathJax 默认脚本通过网络加载。公式暂时显示为源码时，先刷新预览并等待渲染，不要直接把尚未渲染的内容导出。

## 4. 字体：屏幕和打印共用

执行 **Markdown Preview Enhanced: Customize CSS**，打开全局 `style.less`，保留下面的规则，并放在所有 `@media screen` 块之外：

```css
.markdown-preview.markdown-preview {
  font-family: 'Times New Roman', 'SimSun', '宋体', serif;
}
```

之前字体不一致的原因是：宋体规则放在 `@media screen` 中，只作用于预览。Chrome PDF 导出采用打印样式，回退到了主题默认的无衬线字体。

该规则用于正文，不强制替换代码的等宽字体、数学公式字体或图片内的文字。目标电脑需要实际安装相应字体。

## 5. 打印文本：亮色纯黑、暗色纯白

### 5.1 全局打印规则

在 `style.less` 中加入以下规则。已有同样的打印规则时，更新原规则；不要叠加早期“无条件纯黑”的版本。

```css
/* Match PDF text to the rendered preview theme. */
@media print {
  html body .markdown-preview.markdown-preview,
  html body .markdown-preview.markdown-preview *,
  html body .markdown-preview.markdown-preview *::before,
  html body .markdown-preview.markdown-preview *::after {
    color: var(--mpe-print-text-color, #000000) !important;
    -webkit-text-fill-color: var(--mpe-print-text-color, #000000) !important;
    text-shadow: none !important;
  }

  html body .markdown-preview.markdown-preview,
  html body .markdown-preview.markdown-preview *:not(img):not(svg):not(svg *) {
    opacity: 1 !important;
  }

  html body .markdown-preview.markdown-preview svg text,
  html body .markdown-preview.markdown-preview svg tspan {
    fill: var(--mpe-print-text-color, #000000) !important;
    fill-opacity: 1 !important;
    opacity: 1 !important;
  }

  html body .markdown-preview.markdown-preview {
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
  }
}
```

正文、标题、引用、链接、代码文字和 SVG 文本采用相同的纯色。恢复文字所在普通元素的不透明度，是为了避免引用等内容因主题透明度设置而变灰。位图图片不做反色处理，图片内的文字不受这些文本规则控制。

### 5.2 由实际加载的主题提供颜色变量

上述规则必须与主题中的 `--mpe-print-text-color` 配合。变量缺失时会回退为黑色，暗色主题不能仅依靠第 5.1 节自动变成白字。

修改前备份对应主题 CSS。主题文件位于扩展目录下的：

```text
crossnote\styles\preview_theme\
```

对亮色主题，例如 `github-light.css`，在文件末尾添加：

```css
html body { --mpe-print-text-color: #000000; }
```

对暗色主题，例如 `github-dark.css`，在文件末尾添加：

```css
html body { --mpe-print-text-color: #ffffff; }
```

本机已为插件列出的 **17 个主题**添加颜色变量。其中下列 7 个暗色主题使用白色，其他 10 个主题使用黑色：

| 暗色主题 | 正文背景 / PDF 页面背景 |
| --- | --- |
| `atom-dark.css` | `#1d1f21` |
| `atom-material.css` | `#263238` |
| `github-dark.css` | `#24292e` |
| `monokai.css` | `#282828` |
| `night.css` | `#363b40` |
| `one-dark.css` | `#272b33` |
| `solarized-dark.css` | `#002b36` |

将变量放在主题文件中，是为了让插件将主题 CSS 内联到导出 HTML 后仍能确定文字颜色。预览样式中通过 `link[href*="preview_theme/..."]` 判断主题的方案，不能单独覆盖这类内联导出场景。

## 6. 暗色 PDF 外围白框

暗色正文区域之外的白框来自 PDF 的页边距区域。只设置 `body` 的背景不足以覆盖这一区域。

在对应暗色主题 CSS 的末尾加入 `@page` 背景规则。例如 `github-dark.css`：

```css
html body { --mpe-print-text-color: #ffffff; }

/* Fill the PDF page margin without changing content spacing. */
@page { background-color: #24292e; }
```

其他 6 个暗色主题按上一节表格使用对应背景色。本机已全部添加。

这项修改只把页边距区域涂成同一背景色，不把正文推到纸张边缘。0.8.39 的普通 Chrome PDF 导出默认四边为 1 厘米；文档自己的 `chrome` / `puppeteer` front matter 可以覆盖默认导出参数，演示文稿模式也有不同处理。

不要为了消除白框直接把页边距设为 0。当前浏览器环境已验证 `@page` 背景可覆盖纸张外围；其他环境迁移后应重新检查。

## 7. 中文路径图片丢失

### 7.1 原因

示例 Markdown 图片路径：

```markdown
![图 1：Transformer 模型架构](Transformer_译文图像/图1.png)
```

本机 0.8.39 导出时曾把已经编码的路径再次转换为文件 URL，使 `%E8` 等编码变成 `%25E8`。浏览器会尝试读取错误的路径，导致图片丢失。

这属于路径转换问题，单纯延长等待时间无法解决。

### 7.2 本机原生端补丁

文件位于扩展目录中的 `out\native\extension.js`。修改前备份该文件，并确认目标版本中的函数含义一致。当前 0.8.39 的路径转换函数名为 `dd`，最终实现如下（排版展开，逻辑与已安装版本一致）：

```javascript
function dd(e, t) {
  if (Z1e && t) return Z1e(e, t);

  if (e.startsWith("file://")) {
    return e.replace(/^file:\/+/, "file:///");
  }

  try {
    let decoded = decodeURIComponent(e);
    if (!require("fs").existsSync(e) && require("fs").existsSync(decoded)) {
      e = decoded;
    }
  } catch {}

  return (0, R5e.pathToFileURL)(e).href;
}
```

补丁保留已有文件路径；仅在原路径不存在、解码后的路径存在时解码，再生成文件 URL，避免已编码的中文路径被重复编码。

`Z1e`、`R5e` 和 `dd` 都是本版本构建产物中的名称。**不要把这段代码直接替换到其他版本的任意同名位置**；新版本应先检查是否已经修复，以及实际代码结构是否变化。

若可使用 Node.js，修改后执行 `node --check 文件路径` 检查语法，再重新导出验证。此次仅修改原生端文件；不把此补丁当作浏览器版 VS Code 的通用方案。

## 8. 生效、验证与维护

### 应用后

1. 保存设置、全局样式和修改过的扩展文件。
2. 执行 **Developer: Reload Window**（重新加载窗口）。
3. 用 Markdown Preview Enhanced 重新打开 Markdown，等待图片和公式显示完整。
4. 通过 **导出 → Chrome (Puppeteer) → PDF** 重新生成 PDF。
5. 关闭并重新打开 PDF，避免阅读器仍显示旧内容。

### 本次已验证的结果

使用 `Transformer_中文翻译.md` 分别测试了 `github-light.css` 和 `github-dark.css`：

| 检查项 | 验证结果 |
| --- | --- |
| PDF 页数 | 两种主题均为 21 页 |
| 正文嵌入字体 | 包含 SimSun 和 Times New Roman |
| 亮色文本绘制颜色 | RGB `(0, 0, 0)` |
| 暗色文本绘制颜色 | RGB `(255, 255, 255)` |
| 图片 | 5 张均正常加载并写入 PDF |
| 公式 | HTML 导出检查确认 202 处完成 MathJax 渲染 |
| 暗色外围白框 | 对全部 21 页、每页 8 个边缘位置采样，均为 RGB `(36, 41, 46)`，即 `#24292e` |

这些数量对应当时的 Transformer 文档，用于记录验证依据；其他文档不要求具有相同的页数、图片数或公式数。纯色规则不能消除屏幕显示时字体边缘的抗锯齿过渡色。

### 插件更新或迁移后

- 用户 `settings.json` 和 `.crossnote/style.less` 属于用户配置，应单独保留。
- `extension.js`、主题颜色变量和 `@page` 背景属于对扩展安装文件的直接修改，更新或重装可能覆盖，需要核对并重新应用。
- 备份宜放在长期保留的目录；不要把唯一备份放在准备清理的 `tmp` 中。
- 当前工作区的 `tmp/mpe-pdf-fix` 已不存在，不能再依赖以前的脚本、备份路径或测试 PDF。本说明提供了重建核心配置需要的规则。
- 在新电脑上重新确认浏览器路径、字体安装情况、扩展版本，以及文档相对图片路径。

本说明仅整理 PDF 导出相关设置；预览标签去掉 `Preview` 前缀的修改不在本文范围内。
