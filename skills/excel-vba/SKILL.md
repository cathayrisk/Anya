---
name: excel-vba
description: Excel VBA／巨集的撰寫、修改、除錯、效能、安全與交付規範
---

# excel-vba

## 1. 適用範圍與預設值

- 適用：Excel VBA／巨集的撰寫、修改、除錯、審查、效能改善。典型任務是資料整理、
  報表產出、跨活頁簿彙整。
- 不適用：Excel 公式、Power Query、Office Scripts、Word／Access VBA（提醒使用者改問）。
- 預設環境：Windows 64-bit、Excel 2016 以上、`.xlsm`。使用者沒提到 Mac 就不處理 Mac
  差異；不使用 Excel 2019／365 專屬功能。
- **伺服器無法編譯或執行 VBA。** 絕對不可宣稱「已測試通過」「已編譯」「實際跑過」。

## 2. 核心規則

| 等級 | 規則 |
|---|---|
| MUST | 每個模組首行 `Option Explicit`，所有變數明確宣告型別。 |
| MUST | Workbook／Worksheet／Range 一律完整限定；區分 `ThisWorkbook` 與資料活頁簿。 |
| MUST | 發生錯誤時仍要還原 Application 狀態與事件重入旗標（見第 7 節範本）。 |
| MUST | 交付時附模組類型、貼入位置、執行方式、引用需求、相容性（見第 10 節）。 |
| MUST | 不宣稱已編譯或執行；改附 `Debug > Compile VBAProject` 與最小測試步驟。 |
| SHOULD | 用語意命名，不用型別匈牙利前綴。 |
| SHOULD | 大量資料用陣列＋`Value2` 批次讀寫，不逐格存取。 |
| SHOULD | 外部程式庫預設 late binding；若用 early binding 要列出 References 項目。 |
| SHOULD | 處理 `Nothing`、空範圍、`SpecialCells` 找不到、數值溢位。 |
| SHOULD | 缺少非關鍵資訊時直接列出假設，不反覆追問使用者。 |
| AVOID | `Select`、`Activate`、`Selection`、隱含的 `ActiveSheet`／`ActiveWorkbook`。 |
| AVOID | 對大範圍逐格讀寫（每次都是一趟 COM 呼叫）。 |
| AVOID | 全程式範圍的 `On Error Resume Next`；會跳過 cleanup 的 `End`。 |
| AVOID | 列索引用 `Integer`（超過 32,767 列會溢位）；無必要的 `Variant`。 |
| AVOID | 未說明的破壞性操作、外部副作用、版本限定 API。 |

## 3. 動手前要確認或明示的項目

活頁簿與工作表名稱、標題列位置、資料欄位與範圍、資料量級、輸出位置、覆寫政策、
觸發方式（按鈕／`Alt+F8`／事件）、Excel 版本、外部引用。資訊不足時：只有「會實質
改變程式結構」的項目才回頭問（觸發方式、覆寫政策）；其餘在回答開頭以「本次假設」
列出，並讓程式碼在假設不成立時給出清楚錯誤訊息。

## 4. 命名與型別

| 不好 | 好 |
|---|---|
| `strFileName` | `fileName` |
| `intRow` | `rowIndex As Long` |
| `ws` | `sourceSheet` |
| `Sub do_report()` | `Sub BuildRiskReport()` |
| `flag As Boolean` | `hasHeaders As Boolean` |

- Procedure／Class／Enum 用 PascalCase，參數與區域變數用 camelCase，Boolean 用
  `is`／`has`／`can` 開頭，集合用複數。
- 列數、筆數一律 `Long`；日期用 `Date`；金額用 `Currency` 或 `Double`（說明取捨）；
  用 `Enum` 或 `Const` 取代魔術數字與字串常值。
- 修改既有專案時沿用其慣例，不為了改風格而大規模改名。

## 5. 物件限定

| 不好 | 好 |
|---|---|
| `Range("A1")` | `sourceSheet.Range("A1")` |
| `Cells(rowIndex, 1)` | `sourceSheet.Cells(rowIndex, 1)` |
| `ActiveWorkbook` | 明確的 `sourceWorkbook` / `targetWorkbook` |
| `Range("A1:A9").Select` + `Selection.Copy` | `sourceRange.Copy Destination:=targetRange` |
| `Sheets("Data")` | `ThisWorkbook.Worksheets("Data")` |

- 巨集所在檔案用 `ThisWorkbook`，不要假設它就是目前作用中的活頁簿。
- 工作表會被改名：關鍵工作表優先用 CodeName（`Sheet1.Range(...)`），或以名稱查找並在
  找不到時給明確錯誤訊息。`SpecialCells` 找不到儲存格會擲出 Err 1004，必須先攔截。
- 找最後一列用 `sheet.Cells(sheet.Rows.Count, 1).End(xlUp).Row`，不要用 `UsedRange` 推估。

## 6. 效能

| 規則 | 說明 |
|---|---|
| 陣列批次 | 一次讀入二維 `Variant` 陣列，記憶體內處理，一次寫回。 |
| 用 `.Value2` | 比 `.Value`／`.Text` 快，且不會被日期／貨幣格式改變資料。 |
| 減少 COM 往返 | 迴圈內不要重複解析 `Worksheets("x")`，先存成物件變數。 |
| 狀態開關 | 關閉 ScreenUpdating／Calculation／Events，結束一定還原（第 7 節）。 |
| `DoEvents` 節制 | 長迴圈每數百圈呼叫一次避免無回應；每圈都呼叫反而更慢。 |

```vba
' 不好：逐格存取 = 30,000 次 COM 往返
' For rowIndex = 1 To 10000: dataSheet.Cells(rowIndex, 3).Value = ... : Next
' 好：兩次 COM 往返
Dim values As Variant
values = dataSheet.Range("A1:C10000").Value2
For rowIndex = 1 To UBound(values, 1)
    values(rowIndex, 3) = values(rowIndex, 1) * 2
Next rowIndex
dataSheet.Range("A1:C10000").Value2 = values
```

## 7. 狀態還原與事件安全（範本，請沿用此骨架）

```vba
Option Explicit

Private IsRunning As Boolean

Public Sub ProcessRiskData()
    Dim previousScreenUpdating As Boolean, previousDisplayStatusBar As Boolean
    Dim previousEnableEvents As Boolean, previousDisplayPageBreaks As Boolean
    Dim previousCalculation As XlCalculation
    Dim targetSheet As Worksheet
    Dim statesCaptured As Boolean
    Dim errorNumber As Long, errorDescription As String

    If IsRunning Then Exit Sub          ' 事件重入防護
    On Error GoTo CleanFail
    IsRunning = True

    Set targetSheet = ThisWorkbook.Worksheets("Data")

    With Application
        previousScreenUpdating = .ScreenUpdating
        previousDisplayStatusBar = .DisplayStatusBar
        previousCalculation = .Calculation
        previousEnableEvents = .EnableEvents
    End With
    previousDisplayPageBreaks = targetSheet.DisplayPageBreaks
    statesCaptured = True

    With Application
        .ScreenUpdating = False
        .DisplayStatusBar = False
        .Calculation = xlCalculationManual
        .EnableEvents = False
    End With
    targetSheet.DisplayPageBreaks = False

    ' TODO: 主要工作寫在這裡，不使用 Select／Activate。

CleanExit:
    On Error Resume Next
    If statesCaptured Then
        With Application
            .ScreenUpdating = previousScreenUpdating
            .DisplayStatusBar = previousDisplayStatusBar
            .Calculation = previousCalculation
            .EnableEvents = previousEnableEvents
        End With
        targetSheet.DisplayPageBreaks = previousDisplayPageBreaks
    End If
    IsRunning = False
    On Error GoTo 0

    If errorNumber <> 0 Then
        MsgBox "Error " & errorNumber & ": " & errorDescription, _
               vbExclamation, "ProcessRiskData"
    End If
    Exit Sub

CleanFail:
    errorNumber = Err.Number
    errorDescription = Err.Description
    Resume CleanExit
End Sub
```

- 工作表事件（`Worksheet_Change` 等）本體只做判斷與轉呼叫，邏輯放一般模組。
- 事件程序內若會寫入儲存格，必須先關 `Application.EnableEvents` 再於 cleanup 還原。

## 8. 相容性、依賴與安全政策

- 預設 Windows 64-bit Excel 2016+；`Declare` 一律寫 `PtrSafe` 並用 `LongPtr` 型別。
- 預設 late binding（`CreateObject("Scripting.Dictionary")`）；early binding 要列出
  `Tools > References` 需勾選的項目與版本。

**必須先向使用者確認才產出**（說明影響範圍，並建議備份或 dry-run）

1. 清除、刪除或覆寫儲存格、工作表、活頁簿或既有檔案。
2. 批次移動、另存、重新命名或關閉多個檔案（列出目標與衝突處理方式）。
3. 寄信、上傳、列印、寫入資料庫等外部副作用（先預覽對象與筆數）。
4. `Shell`、系統 API、登錄檔、修改 VBProject（說明命令、風險與回復方式）。

**直接拒絕，並說明原因**

5. 關閉或繞過 Trust Center、Protected View、數位簽章、防毒／EDR。
6. 破解、猜測或繞過活頁簿、工作表、VBA Project、檔案或帳號密碼。
7. 蒐集憑證、token 或敏感資料並隱密寄送、上傳、外洩。
8. 下載執行 payload、隱匿持久化、規避偵測、自我複製、蓄意破壞資料。

## 9. 交付前靜態自檢（逐條核對後才輸出）

1. 每個模組有 `Option Explicit`，沒有未宣告或未使用的變數。
2. `Sub`／`Function`／`If`／`For`／`With`／`Select Case` 區塊全部成對收尾。
3. Range／Cells／Worksheets 都有限定父物件；沒有非必要的 `Select`／`Activate`／`End`／
   全域 `On Error Resume Next`。
4. 每個被改動的 Application 狀態都在 cleanup 還原成原值（含錯誤路徑）。
5. 已處理 `Nothing`、空範圍、`SpecialCells` 無結果、除以零、`Long` 溢位。
6. 引用需求、Excel 版本、32／64-bit 差異已交代；無第 8 節的高風險操作或已附確認機制。

`run_python` 只能驗證 Python 邏輯，不是 VBA 的實測，不可拿來當「VBA 測過了」的證據。

## 10. 交付格式

每次交付至少包含：

1. 功能摘要與本次採用的假設。
2. 完整程式碼（不可用省略號略過任何段落）。
3. 模組類型（標準／類別／`ThisWorkbook`／特定工作表／UserForm）、建議名稱與貼入位置，
   多模組標示順序；執行方式（`Alt+F8`／按鈕／事件／工作表函式）。
4. 引用需求（late binding 則註明「不需額外引用」）、`.xlsm` 存檔提醒、巨集安全性設定、
   Excel 版本需求。
5. 已知限制與風險（含破壞性操作的確認機制）。
6. 手動測試案例，至少涵蓋正常、空資料、異常資料三種。
7. 明確聲明「本程式碼未在 Excel／VBE 實際編譯或執行，請先執行
   `Debug > Compile VBAProject` 再以小量資料試跑」。

## 11. 來源（均為摘要改寫，本 skill 不依賴任何外部檔案）

- Microsoft Learn, Excel performance: Tips for optimizing performance obstructions
  （Create faster VBA macros 一節）與 Excel object model reference。
- RenRMT/claude-code-vba-skills（MIT License）：架構、命名、物件模型、錯誤處理。
- Govert van Drimmelen, Excel/VBA Agentic Coding Guide（2026-02）：編譯錯誤與溢位陷阱。
