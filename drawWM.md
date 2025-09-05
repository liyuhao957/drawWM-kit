# 水印绘制逻辑说明（drawWMC / $utils.drawWM）

## 绘制流程（$utils.drawWM）
1. 计算画布尺寸
   - 逻辑宽度固定为 `width = 750`。
   - 高度来自 `getPageHeight(ctx)`：根据设备窗口尺寸按设计宽 750 做等比换算。

2. 获取画布上下文
   - `const canvas = ctx.$element(cvId)`；如未取到直接返回。
   - `const cvctx = canvas.getContext('2d')`。
   - 先 `cvctx.clearRect(0, 0, width, height)` 清空。
   - 若 `clear === true`，到此结束（实现“清屏”）。

3. 生成编码矩阵（pattern → 二进制位阵列）
   - 数据源：
     - `const { userId = 'unknownUserId' } = getQAIDs()` 取用户 ID。
     - `tf('DDHHmm')` 取当前日期时分（“日+小时+分钟”），与 `userId` 的后 6 位拼成 12 字符串 `pattern`。
   - 字符到数字：
     - 对每个字符 `c`：`cc = c.charCodeAt(0)`；
       - 若 `cc > 57`（大于 '9'），`num = cc - 87`（'a' → 10，'z' → 35）
       - 否则 `num = cc - 48`（'0' → 0，'9' → 9）
   - 数字到二进制位：
     - `num.toString(2)` 得二进制字符串 → `split('')` → `map(Number)` → `reverse()`
     - 注意：未做固定位数（6 位）补零；越高位越可能为空，位阵呈“下密上疏”。

4. 网格与布局（点阵平铺）
   - 基本尺寸：
     - `ratio = 10`（每个点阵小方块边长像素）
     - `gap = 7`（小方块间距）
     - 单个“码单元 cell”包含 12 列 x 6 行位点：
       - `cellWidth  = 12 * (ratio + gap) - gap`
       - `cellHeight =  6 * (ratio + gap) - gap`
   - 平铺布局：
     - 列数 `cols = 3`，行数 `rows = 11`
     - 横向间距：`colGap = floor((width  - cellWidth  * cols) / (cols + 1))`
     - 纵向间距：`rowGap = floor((height - cellHeight * rows) / (rows + 1))`
   - 每个 cell 的起点：
     - `xOffset = j * (cellWidth + colGap) + colGap`
     - `yOffset = i * (cellHeight + rowGap) + rowGap`

5. 实际绘制
   - 遍历 11x3 个 cell；对每个 cell：
     - 遍历 `matrixArr`（12 个字符 → 12 列）：列内 X 坐标 `x = idx * (ratio + gap) + xOffset`
     - 遍历该列的二进制位数组 `nums`：当 `flag === 1` 时，计算 Y：`y = idx2 * (ratio + gap) + yOffset`
     - `rsize = ratio / 2`，对一个位点绘制 2x2 子方块棋盘格：
       - `fillStyle = 'rgb(5, 5, 5)'`，填充 `(x, y, rsize, rsize)` 与 `(x + rsize, y + rsize, rsize, rsize)`
       - `fillStyle = 'rgb(15, 15, 15)'`，填充 `(x + rsize, y, rsize, rsize)` 与 `(x, y + rsize, rsize, rsize)`
       - 每次绘制前后 `cvctx.save()`/`cvctx.restore()`（源码一次 save/restore 包裹同一位点的四次填充）

## 视觉与行为特性
- 单个位点以 2x2 子方块棋盘格绘制：使用 `rgb(5,5,5)` 与 `rgb(15,15,15)` 两种近黑色，提高细微对比度（在深色背景上更易感知）。
- 单元重复平铺，既保证覆盖密度，又避免单次绘制的过多像素（性能轻量）。
- 每次调用先清屏确保无残影，`clear` 参数控制“是绘制还是仅清空”。

## 适配与设备差异
- `getPageHeight(ctx)`：
  - 非小米设备数据来自 `ctx.$page`；小米品牌走 `getDeviceInfo()`。
  - 返回按 750 设计宽换算后的 `windowHeight`、`statusBarHeight`。水印高度使用换算后的 `windowHeight`。
- 若不同机型上 `windowWidth/Height` 或 `statusBarHeight` 获取异常，可能影响间距计算与铺排效果。

## 常见问题排查
- 看不到水印：
  - `wmable` 是否为真？模板里 `<canvas id="wmc" if="{{wmable}}">` 是否渲染？
  - `layerMaterial` 是否为空？在 `drawWMC` 中为空会导致只清屏不绘制。
  - 叠层遮挡：检查蒙层/图片/组件的层级与透明度，确认 `<canvas>` 未被完全覆盖。
- 闪烁/频繁重绘：
  - `layerMaterial` 频繁变化会导致重复清屏与绘制；必要时可在 `drawWMC` 做节流/去抖。
- 可视性差：
  - 背景过深时，可考虑调整填充色或透明度，或在 UI 上提高对比度。

## 可选改进建议
- 使用 `requestAnimationFrame` 替换 `setTimeout(0)`，更贴合下一帧绘制时机。
- 为二进制位统一到固定高度（例如 6 位左侧补零），让点阵上下对齐更规整。
- 将颜色、透明度、平铺行列数做成可配置项，适配不同背景与画布比例。

## 快速参考（关键参数）
- 画布宽度：`750`
- 小方块：`ratio = 10`，`gap = 7`，`rsize = ratio / 2`
- 单元大小：`12 x 6` 位点
- 平铺：`cols = 3`，`rows = 11`
- 清屏逻辑：`clearRect → if (clear) return`
- pattern：`userId.slice(-6) + tf('DDHHmm')` → 12 字符 → [0-9a-z]→[0..35]→二进制位阵列（反转）

