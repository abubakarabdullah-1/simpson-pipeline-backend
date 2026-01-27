from openpyxl import Workbook

#.......................................
def create_excel_from_result(result: dict, output_path: str):

    wb = Workbook()

    line_items = result.get("line_items", [])
    confidence = result.get("confidence")
    grand_total = result.get("grand_total")
    survey_data = result.get("survey_data", {})
    project_specs = result.get("project_specs", {})

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


    # -------------------------------
    # Fallback: Check for missing specs from Survey Data
    # -------------------------------
    # Create a set of processed items (Page, View, clean_tag) to avoid duplicates
    processed_keys = set()
    for item in line_items:
        if str(item.get("Category", "")).strip().lower() == "deduction":
            # Extract tag from description "Opening TYPE A" -> "TYPE A"
            desc = item.get("Description", "")
            # Simple heuristic: assume description ends with the tag or contains it
            processed_keys.add(f"{item.get('Page')}_{item.get('View')}_{desc}")

    for page_num, views in survey_data.items():
        for view_label, tags in views.items():
             for tag, count in tags.items():
                  # Clean tag logic from phase5_v2 reuse
                  clean_tag = str(tag).upper().replace("TYPE", "").strip()
                  
                  # Check if we have a robust match in processed_keys
                  # Since we can't easily reconstruct the exact Description phase5 used, 
                  # we'll look for fuzzy match or just see if "Opening" + Clean Tag is plausible
                  
                  # Actually, easier: check if this tag exists in the specs. 
                  # If NOT, it's definitely missing.
                  
                  # Flatten library (lite version of phase5 logic)
                  all_types = {}
                  all_types.update(project_specs.get('windows', {}))
                  all_types.update(project_specs.get('doors', {}))
                  
                  spec_found = False
                  if clean_tag in all_types:
                      spec_found = True
                  else:
                      for key in all_types:
                          if clean_tag == key.replace("-", "") or key == clean_tag.replace("-", ""):
                              spec_found = True
                              break
                  
                  if not spec_found:
                      ws_ded.append([
                        page_num,
                        view_label,
                        f"Opening {clean_tag} [Missing Spec]",
                        count,
                        0.0,
                        0.0
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
