import re


# ============================================================
# BASIC TEXT CLEANING
# ============================================================

def clean_text(text):
    if text is None:
        return ""

    text = str(text)

    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n+", "\n", text)

    return text.strip()


def clean_value(value):
    if value is None:
        return ""

    value = str(value)
    value = re.sub(r"\s+", " ", value)
    value = value.strip(" :|-")

    return value.strip()


def build_combined_text(words):
    if not words:
        return ""

    return clean_text(" ".join(str(word) for word in words if word))


# ============================================================
# DOCUMENT TYPE DETECTION
# ============================================================

def detect_document_type(text):
    text_lower = text.lower()

    # --------------------------------------------------------
    # Academic documents
    # --------------------------------------------------------

    academic_patterns = [
        "grade card",
        "marks memo",
        "marksheet",
        "mark sheet",
        "transcript",
        "semester",
        "sgpa",
        "cgpa",
        "course code",
        "credits",
        "grade",
        "b.tech degree examination",
        "degree examination",
        "academic year",
        "result",
    ]

    academic_score = sum(
        1
        for pattern in academic_patterns
        if pattern in text_lower
    )

    if academic_score >= 2:
        return "Academic Document"

    # --------------------------------------------------------
    # Receipts / invoices
    # --------------------------------------------------------

    receipt_patterns = [
        "receipt",
        "fee receipt",
        "payment receipt",
        "transaction id",
        "txn id",
        "amount paid",
        "payment mode",
        "total amount",
        "invoice",
        "invoice no",
        "table no",
        "table no.",
        "cashier",
        "subtotal",
        "total",
        "clerk",
    ]

    receipt_score = sum(
        1
        for pattern in receipt_patterns
        if pattern in text_lower
    )

    currency_receipt = bool(
        re.search(
            r"(?:£|€|\$|₹|rs\.?|inr)\s*\d",
            text,
            re.IGNORECASE
        )
    )

    food_receipt_terms = [
        "soft drink",
        "cod & chips",
        "cod&chips",
        "cod and chips",
        "bread & butter",
        "bread&butter",
        "bread and butter",
    ]

    food_score = sum(
        1
        for pattern in food_receipt_terms
        if pattern in text_lower
    )

    if (
        receipt_score >= 2
        or currency_receipt
        or food_score >= 2
    ):
        return "Receipt / Invoice"

    # --------------------------------------------------------
    # Identity documents
    # --------------------------------------------------------

    identity_patterns = [
        "aadhaar",
        "pan card",
        "passport",
        "voter id",
        "identity card",
        "id card",
        "unique disability id",
    ]

    if any(
        pattern in text_lower
        for pattern in identity_patterns
    ):
        return "Identity Document"

    # --------------------------------------------------------
    # Certificates
    # --------------------------------------------------------

    certificate_patterns = [
        "certificate",
        "certify that",
        "this is to certify",
        "community certificate",
        "disability certificate",
        "income certificate",
        "bonafide certificate",
    ]

    if any(
        pattern in text_lower
        for pattern in certificate_patterns
    ):
        return "Certificate"

    # --------------------------------------------------------
    # Forms / applications
    # --------------------------------------------------------

    form_patterns = [
        "application form",
        "application no",
        "applicant",
        "date of birth",
        "father name",
        "mother name",
        "address",
    ]

    form_score = sum(
        1
        for pattern in form_patterns
        if pattern in text_lower
    )

    if form_score >= 3:
        return "Application/Form"

    return "Unknown Document"


# ============================================================
# REGULAR EXPRESSION HELPERS
# ============================================================

def extract_first(patterns, text, flags=re.IGNORECASE):
    for pattern in patterns:
        match = re.search(pattern, text, flags)
        if match:
            value = match.group(1)
            value = clean_value(value)

            if value:
                return value

    return ""


# ============================================================
# DATES
# ============================================================

def extract_dates(text):
    patterns = [
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        r"\b\d{1,2}[/-][A-Za-z]{3,9}[/-]\d{2,4}\b",
        r"\b\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}\b",
        r"\b[A-Za-z]{3,9}\s+\d{1,2},\s+\d{4}\b",
        r"\b\d{1,2}\s+[A-Za-z]{3,9},\s+\d{4}\b",
    ]

    dates = []

    for pattern in patterns:
        matches = re.findall(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        for match in matches:
            value = clean_value(match)

            if value and value not in dates:
                dates.append(value)

    return dates


# ============================================================
# EMAILS
# ============================================================

def extract_emails(text):
    pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"

    return list(
        dict.fromkeys(
            re.findall(pattern, text)
        )
    )


# ============================================================
# PHONE NUMBERS
# ============================================================

def extract_phones(text):
    patterns = [
        r"\+?\d[\d\s().-]{7,}\d",
    ]

    phones = []

    for pattern in patterns:
        matches = re.findall(
            pattern,
            text
        )

        for match in matches:
            value = clean_value(match)

            if len(re.sub(r"\D", "", value)) >= 8:
                if value not in phones:
                    phones.append(value)

    return phones


# ============================================================
# URLS
# ============================================================

def extract_urls(text):
    pattern = (
        r"\b(?:https?://|www\.)"
        r"[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
        r"(?:/[^\s]*)?"
    )

    return list(
        dict.fromkeys(
            re.findall(
                pattern,
                text,
                flags=re.IGNORECASE
            )
        )
    )


# ============================================================
# REFERENCE NUMBERS
# ============================================================

def extract_reference_numbers(text):
    patterns = [
        r"\b(?:ref(?:erence)?\.?\s*(?:no|number)?|"
        r"application\s*(?:no|number)|"
        r"certificate\s*(?:no|number)|"
        r"invoice\s*(?:no|number)|"
        r"transaction\s*(?:id|no|number)|"
        r"txn\s*(?:id|no|number))"
        r"\s*[:#-]?\s*([A-Z0-9/-]{4,})\b",
    ]

    values = []

    for pattern in patterns:
        matches = re.findall(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        for match in matches:
            value = clean_value(match)

            if value and value not in values:
                values.append(value)

    return values


# ============================================================
# AMOUNTS
# ============================================================

def extract_amounts(text):
    pattern = (
        r"(?:₹|Rs\.?|INR|\$|€|£)"
        r"\s*"
        r"\d+(?:,\d{3})*(?:\.\d{1,2})?"
        r"|"
        r"\b\d+(?:,\d{3})*(?:\.\d{1,2})?"
        r"\s*(?:₹|Rs\.?|INR|\$|€|£)"
    )

    return list(
        dict.fromkeys(
            clean_value(match)
            for match in re.findall(
                pattern,
                text,
                flags=re.IGNORECASE
            )
        )
    )


# ============================================================
# FIELD ALIASES
# ============================================================

FIELD_ALIASES = {
    "name": [
        "name",
        "student name",
        "candidate name",
        "applicant name",
        "full name",
        "student",
    ],
    "registration_number": [
        "registration number",
        "registration no",
        "registration id",
        "reg no",
        "reg number",
        "roll number",
        "roll no",
        "student id",
    ],
    "course": [
        "course",
        "degree",
        "program",
        "programme",
    ],
    "branch": [
        "branch",
        "specialization",
        "specialisation",
        "department",
    ],
    "semester": [
        "semester",
        "sem",
    ],
    "sgpa": [
        "sgpa",
    ],
    "cgpa": [
        "cgpa",
    ],
    "date": [
        "date",
        "date of birth",
        "issue date",
        "printed on",
        "issued on",
    ],
    "institution": [
        "institution",
        "college",
        "university",
        "school",
    ],
    "certificate_number": [
        "certificate number",
        "certificate no",
        "certificate id",
    ],
}


# ============================================================
# LABEL-VALUE EXTRACTION
# ============================================================

def extract_label_values(text):
    fields = {}

    lines = [
        clean_value(line)
        for line in text.splitlines()
        if clean_value(line)
    ]

    for canonical_name, aliases in FIELD_ALIASES.items():

        for line in lines:

            for alias in aliases:

                pattern = (
                    r"^"
                    + re.escape(alias)
                    + r"\s*[:#-]?\s*(.+)$"
                )

                match = re.match(
                    pattern,
                    line,
                    flags=re.IGNORECASE
                )

                if match:
                    value = clean_value(match.group(1))

                    if value:
                        fields[canonical_name] = value
                        break

            if canonical_name in fields:
                break

    return fields


# ============================================================
# GRADE CARD DETECTION
# ============================================================

def is_grade_card(text):
    text_lower = text.lower()

    indicators = [
        "grade card",
        "course code",
        "sgpa",
        "cgpa",
        "credits",
        "name of the course",
    ]

    score = sum(
        1
        for indicator in indicators
        if indicator in text_lower
    )

    return score >= 3


# ============================================================
# GRADE CARD FIELDS
# ============================================================

def extract_grade_card_fields(text):
    fields = {}

    name = extract_first(
        [
            r"(?:student\s+name|name)\s*[:\-]\s*(.+)",
        ],
        text
    )

    if name:
        fields["Name"] = name

    registration_number = extract_first(
        [
            r"(?:registration\s*(?:number|no|id)|"
            r"reg\s*(?:no|number))\s*[:\-]?\s*([A-Z0-9/-]+)"
        ],
        text
    )

    if registration_number:
        fields["Registration Number"] = registration_number

    course = extract_first(
        [
            r"(?:course|degree|program|programme)\s*[:\-]\s*(.+)"
        ],
        text
    )

    if course:
        fields["Course Degree"] = course

    branch = extract_first(
        [
            r"(?:branch|specialization|specialisation|department)"
            r"\s*[:\-]\s*(.+)"
        ],
        text
    )

    if branch:
        fields["Branch"] = branch

    semester = extract_first(
        [
            r"(?:semester|sem)\s*[:\-]?\s*(.+)"
        ],
        text
    )

    if semester:
        fields["Semester"] = semester

    sgpa = extract_first(
        [
            r"\bsgpa\s*[:\-]?\s*(\d+(?:\.\d+)?)"
        ],
        text
    )

    if sgpa:
        fields["Sgpa"] = sgpa

    cgpa = extract_first(
        [
            r"\bcgpa\s*[:\-]?\s*(\d+(?:\.\d+)?)"
        ],
        text
    )

    if cgpa:
        fields["Cgpa"] = cgpa

    printed_date = extract_first(
        [
            r"printed\s*(?:on)?\s*[:\-]?\s*"
            r"(\d{1,2}[-/][A-Za-z]{3,9}[-/]\d{2,4})",
            r"printed\s*(?:on)?\s*[:\-]?\s*"
            r"(\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})",
        ],
        text
    )

    if printed_date:
        fields["Printed Date"] = printed_date

    institution = extract_first(
        [
            r"(?:institution|university|college)\s*[:\-]\s*(.+)"
        ],
        text
    )

    if institution:
        fields["Institution"] = institution

    return fields


# ============================================================
# GRADE CARD TABLE
# ============================================================

def extract_grade_card_table(text):
    rows = []

    text = clean_text(text)

    grade_pattern = (
        r"(?:O|A\+|A-|A|B\+|B-|B|C\+|C-|C|"
        r"D\+|D-|D|E|F|P|S)"
    )

    course_code_pattern = (
        r"\b(?:"
        r"\d{2}[A-Z]{2,6}\d{3,5}"
        r"|"
        r"[A-Z]{3,6}\d{3,5}"
        r")\b"
    )

    matches = list(
        re.finditer(
            course_code_pattern,
            text,
            flags=re.IGNORECASE
        )
    )

    if not matches:
        return rows

    for index, match in enumerate(matches):

        course_code = clean_text(
            match.group(0)
        )

        start = match.end()

        if index + 1 < len(matches):
            end = matches[index + 1].start()
        else:
            end = len(text)

        segment = clean_text(
            text[start:end]
        )

        segment = re.split(
            r"\b(?:SGPA|CGPA|Printed\s+On|Note)\b",
            segment,
            maxsplit=1,
            flags=re.IGNORECASE
        )[0]

        segment = clean_text(segment)

        if not segment:
            continue

        tail_pattern = (
            r"^(.*?)"
            r"\s+"
            r"(\d+(?:\.\d+)?)"
            r"\s+("
            + grade_pattern
            + r")"
            r"\s*$"
        )

        tail_match = re.match(
            tail_pattern,
            segment,
            flags=re.IGNORECASE
        )

        if not tail_match:
            continue

        course_name = clean_text(
            tail_match.group(1)
        )

        credits = clean_text(
            tail_match.group(2)
        )

        grade = clean_text(
            tail_match.group(3)
        ).upper()

        if (
            not course_name
            or len(course_name) < 2
            or len(course_name) > 150
        ):
            continue

        invalid_names = {
            "name of the course",
            "course code",
            "credits",
            "grade",
            "sgpa",
            "cgpa",
        }

        if course_name.lower() in invalid_names:
            continue

        rows.append(
            [
                course_code,
                course_name,
                credits,
                grade,
            ]
        )

    unique_rows = []

    seen = set()

    for row in rows:

        key = tuple(row)

        if key not in seen:
            seen.add(key)
            unique_rows.append(row)

    return unique_rows


# ============================================================
# RECEIPT / INVOICE EXTRACTION
# ============================================================

def extract_receipt_fields(text):
    fields = {}

    text_clean = clean_text(text)

    # Total amount
    total_patterns = [
        r"(?:grand\s+total|total\s+amount|amount\s+paid|"
        r"total)\s*[:\-]?\s*"
        r"(?:£|€|\$|₹|rs\.?|inr)?\s*"
        r"(\d+(?:\.\d{1,2})?)",

        r"(?:£|€|\$|₹|rs\.?|inr)\s*"
        r"(\d+(?:\.\d{1,2})?)\s*$",
    ]

    total = extract_first(
        total_patterns,
        text_clean
    )

    if total:
        fields["Total Amount"] = total

    # Transaction ID
    transaction_id = extract_first(
        [
            r"(?:transaction\s*(?:id|no|number)|txn\s*(?:id|no|number))"
            r"\s*[:#-]?\s*([A-Z0-9/-]+)"
        ],
        text_clean
    )

    if transaction_id:
        fields["Transaction ID"] = transaction_id

    # Invoice number
    invoice_number = extract_first(
        [
            r"(?:invoice\s*(?:no|number|id))"
            r"\s*[:#-]?\s*([A-Z0-9/-]+)"
        ],
        text_clean
    )

    if invoice_number:
        fields["Invoice Number"] = invoice_number

    # Table number
    table_number = extract_first(
        [
            r"(?:table\s*(?:no|number))"
            r"\s*[:#-]?\s*([A-Z0-9/-]+)"
        ],
        text_clean
    )

    if table_number:
        fields["Table Number"] = table_number

    # Date
    dates = extract_dates(text_clean)

    if dates:
        fields["Date"] = dates[0]

    # Phone
    phones = extract_phones(text_clean)

    if phones:
        fields["Phone"] = phones[0]

    # Currency amounts
    amounts = extract_amounts(text_clean)

    if amounts:
        fields["Amounts"] = ", ".join(amounts)

    # Payment mode
    payment_mode = extract_first(
        [
            r"(?:payment\s*mode|mode\s*of\s*payment)"
            r"\s*[:\-]?\s*([A-Za-z ]+)"
        ],
        text_clean
    )

    if payment_mode:
        fields["Payment Mode"] = payment_mode

    # Cashier / clerk
    cashier = extract_first(
        [
            r"(?:cashier|clerk)\s*[:#-]?\s*([A-Za-z0-9]+)"
        ],
        text_clean
    )

    if cashier:
        fields["Cashier / Clerk"] = cashier

    return fields


# ============================================================
# CERTIFICATE EXTRACTION
# ============================================================

def extract_certificate_fields(text):
    fields = {}

    name = extract_first(
        [
            r"(?:name|awarded\s+to|presented\s+to)"
            r"\s*[:\-]?\s*([A-Za-z][A-Za-z .'-]{2,100})"
        ],
        text
    )

    if name:
        fields["Name"] = name

    certificate_number = extract_first(
        [
            r"(?:certificate\s*(?:no|number|id))"
            r"\s*[:#-]?\s*([A-Z0-9/-]+)"
        ],
        text
    )

    if certificate_number:
        fields["Certificate Number"] = certificate_number

    institution = extract_first(
        [
            r"(?:institution|university|college|school)"
            r"\s*[:\-]\s*(.+)"
        ],
        text
    )

    if institution:
        fields["Institution"] = institution

    dates = extract_dates(text)

    if dates:
        fields["Date"] = dates[0]

    return fields


# ============================================================
# ACADEMIC INFORMATION
# ============================================================

def extract_academic_information(text):
    information = {}

    dates = extract_dates(text)

    if dates:
        information["Dates"] = dates

    emails = extract_emails(text)

    if emails:
        information["Emails"] = emails

    phones = extract_phones(text)

    if phones:
        information["Phone Numbers"] = phones

    urls = extract_urls(text)

    if urls:
        information["URLs"] = urls

    references = extract_reference_numbers(text)

    if references:
        information["Reference Numbers"] = references

    amounts = extract_amounts(text)

    if amounts:
        information["Amounts"] = amounts

    return information


# ============================================================
# ACADEMIC FIELDS
# ============================================================

def extract_academic_fields(text):
    fields = {}

    grade_fields = extract_grade_card_fields(text)

    if grade_fields:
        fields.update(grade_fields)

    label_fields = extract_label_values(text)

    for key, value in label_fields.items():

        if key not in fields:
            fields[key] = value

    return fields


# ============================================================
# GENERIC TABLE DATA
# ============================================================

def extract_table_data(text, document_type=""):
    rows = []

    if document_type == "Academic Document":
        return extract_grade_card_table(text)

    lines = [
        clean_value(line)
        for line in text.splitlines()
        if clean_value(line)
    ]

    for line in lines:

        if "|" in line:

            parts = [
                clean_value(part)
                for part in line.split("|")
            ]

            parts = [
                part
                for part in parts
                if part
            ]

            if len(parts) >= 2:
                rows.append(parts)

        elif "\t" in line:

            parts = [
                clean_value(part)
                for part in line.split("\t")
            ]

            parts = [
                part
                for part in parts
                if part
            ]

            if len(parts) >= 2:
                rows.append(parts)

    return rows


# ============================================================
# NUMBERED ENTRIES
# ============================================================

def extract_numbered_entries(text):
    entries = []

    pattern = r"(?m)^\s*(\d+)[.)]\s*(.+)$"

    matches = re.findall(
        pattern,
        text
    )

    for number, value in matches:

        value = clean_value(value)

        if value:
            entries.append(
                {
                    "number": number,
                    "value": value,
                }
            )

    return entries


# ============================================================
# GENERIC ENTITIES
# ============================================================

def extract_generic_entities(text):
    entities = {}

    dates = extract_dates(text)

    if dates:
        entities["dates"] = dates

    emails = extract_emails(text)

    if emails:
        entities["emails"] = emails

    phones = extract_phones(text)

    if phones:
        entities["phones"] = phones

    urls = extract_urls(text)

    if urls:
        entities["urls"] = urls

    references = extract_reference_numbers(text)

    if references:
        entities["reference_numbers"] = references

    amounts = extract_amounts(text)

    if amounts:
        entities["amounts"] = amounts

    return entities


# ============================================================
# MERGE FIELDS
# ============================================================

def merge_fields(*field_sets):
    merged = {}

    for field_set in field_sets:

        if not field_set:
            continue

        for key, value in field_set.items():

            if value is None:
                continue

            if isinstance(value, str):
                value = clean_value(value)

                if not value:
                    continue

            merged[key] = value

    return merged


# ============================================================
# MAIN DOCUMENT EXTRACTION
# ============================================================

def extract_document_info(
    text=None,
    words=None,
    boxes=None
):
    if text is None:
        text = build_combined_text(words or [])

    text = clean_text(text)

    document_type = detect_document_type(text)

    entities = extract_generic_entities(text)

    structured_fields = {}

    label_values = extract_label_values(text)

    structured_fields = merge_fields(
        structured_fields,
        label_values
    )

    # --------------------------------------------------------
    # Academic documents
    # --------------------------------------------------------

    if document_type == "Academic Document":

        academic_fields = extract_academic_fields(
            text
        )

        structured_fields = merge_fields(
            structured_fields,
            academic_fields
        )

        academic_information = extract_academic_information(
            text
        )

        entities["academic_information"] = (
            academic_information
        )

        if is_grade_card(text):

            grade_card_fields = extract_grade_card_fields(
                text
            )

            structured_fields = merge_fields(
                structured_fields,
                grade_card_fields
            )

    # --------------------------------------------------------
    # Receipts / invoices
    # --------------------------------------------------------

    elif document_type == "Receipt / Invoice":

        receipt_fields = extract_receipt_fields(
            text
        )

        structured_fields = merge_fields(
            structured_fields,
            receipt_fields
        )

    # --------------------------------------------------------
    # Certificates
    # --------------------------------------------------------

    elif document_type == "Certificate":

        certificate_fields = extract_certificate_fields(
            text
        )

        structured_fields = merge_fields(
            structured_fields,
            certificate_fields
        )

    # --------------------------------------------------------
    # Other documents
    # --------------------------------------------------------

    else:

        academic_information = extract_academic_information(
            text
        )

        if academic_information:
            entities["academic_information"] = (
                academic_information
            )

    # --------------------------------------------------------
    # Tables
    # --------------------------------------------------------

    table_data = extract_table_data(
        text,
        document_type
    )

    # --------------------------------------------------------
    # Numbered entries
    # --------------------------------------------------------

    numbered_entries = extract_numbered_entries(
        text
    )

    # --------------------------------------------------------
    # Final entities
    # --------------------------------------------------------

    entities["label_values"] = structured_fields

    if table_data:
        entities["table_data"] = table_data

    if numbered_entries:
        entities["numbered_entries"] = numbered_entries

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    document_information = {
        "Document Type": document_type,
        "Fields": structured_fields,
    }

    return {
        "document_information": document_information,
        "document_type": document_type,
        "text": text,
        "entities": entities,
    }