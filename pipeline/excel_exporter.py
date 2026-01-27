from openpyxl import Workbook

#.......................................
def create_excel_from_result(result: dict, output_path: str):

    wb = Workbook()

    line_items = result.get("line_items", [])
    confidence = result.get("confidence")
    grand_total = result.get("grand_total")

    # -------------------------------
    # Calculate Total Deductions
    # -------------------------------
    total_deductions = sum(
        item.get("Total_SF", 0)
        for item in line_items
        if str(item.get("Category", "")).strip().lower() == "deduction"
    )

    # ===============================
    # SUMMARY SHEET
    # ===============================
    ws_summary = wb.active
    ws_summary.title = "Summary"

    ws_summary.append(["Metric", "Value"])
    ws_summary.append(["Grand Total SF", grand_total])
    ws_summary.append(["Total Deductions SF", total_deductions])
    ws_summary.append(["Confidence", confidence])

    # ===============================
    # DEDUCTIONS SHEET
    # ===============================
    ws_ded = wb.create_sheet("Deductions")

    ws_ded.append([
        "Page",
        "View",
        "Description",
        "EA",
        "Unit_SF",
        "Total_SF",
    ])

    for item in line_items:
        if str(item.get("Category", "")).strip().lower() == "deduction":
            ws_ded.append([
                item.get("Page"),
                item.get("View"),
                item.get("Description"),
                item.get("Count"),
                item.get("Unit_SF"),
                item.get("Total_SF"),
            ])

    # ===============================
    # EIFS SHEET
    # ===============================
    ws_eifs = wb.create_sheet("EIFS")

    ws_eifs.append([
        "Page",
        "View",
        "Description",
        "EA",
        "Unit_SF",
        "Total_SF",
    ])

    for item in line_items:
        category = item.get("Category", "")

        if "EIFS" in category.upper():
            ws_eifs.append([
                item.get("Page"),
                item.get("View"),
                item.get("Description"),
                item.get("Count"),
                item.get("Unit_SF"),
                item.get("Total_SF"),
            ])

    wb.save(output_path)
