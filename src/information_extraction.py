import re
from collections import OrderedDict


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    if text is None:
        return ""

    text = str(text)

    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    text = text.replace("\t", " ")

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def build_combined_text(words):
    cleaned_words = []

    for word in words or []:
        value = clean_text(word)

        if value:
            cleaned_words.append(value)

    return " ".join(cleaned_words)


# ============================================================
# DOCUMENT TYPE DETECTION
# ============================================================

def detect_document_type(text):
    text_lower = text.lower()

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
    ]

    receipt_score = sum(
        1
        for pattern in receipt_patterns
        if pattern in text_lower
    )

    if receipt_score >= 2:
        return "Receipt / Invoice"

    identity_patterns = [
        "aadhaar",
        "pan card",
        "passport",
        "voter id",
        "identity card",
        "id card",
        "unique disability id",
    ]

    if any(pattern in text_lower for pattern in identity_patterns):
        return "Identity Document"

    certificate_patterns = [
        "certificate",
        "certify that",
        "this is to certify",
        "community certificate",
        "disability certificate",
        "income certificate",
        "bonafide certificate",
    ]

    if any(pattern in text_lower for pattern in certificate_patterns):
        return "Certificate"

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
# BASIC ENTITY EXTRACTION
# ============================================================

def extract_dates(text):
    patterns = [
        r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b",
        r"\b\d{1,2}[-/][A-Za-z]{3,9}[-/]\d{2,4}\b",
        r"\b\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4}\b",
        r"\b[A-Za-z]{3,9}\s+\d{4}\b",
    ]

    results = []

    for pattern in patterns:
        for match in re.findall(
            pattern,
            text,
            flags=re.IGNORECASE
        ):
            if match not in results:
                results.append(match)

    return results


def extract_emails(text):
    return list(
        OrderedDict.fromkeys(
            re.findall(
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
                text
            )
        )
    )


def extract_phone_numbers(text):
    numbers = re.findall(
        r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)",
        text
    )

    return list(
        OrderedDict.fromkeys(numbers)
    )


def extract_urls(text):
    urls = re.findall(
        r"\bhttps?://[^\s]+",
        text,
        flags=re.IGNORECASE
    )

    return list(
        OrderedDict.fromkeys(urls)
    )


# ============================================================
# REFERENCE NUMBER EXTRACTION
# ============================================================

def extract_reference_numbers(text):
    results = []

    patterns = [
        r"\b(?:Regd\.?\s*No\.?|Registration\s*No\.?|Register\s*No\.?)\s*[:\-]?\s*([A-Za-z0-9/-]+)",
        r"\b(?:Application\s*No\.?|Application\s*Number)\s*[:\-]?\s*([A-Za-z0-9/-]+)",
        r"\b(?:Reference\s*No\.?|Reference\s*Number)\s*[:\-]?\s*([A-Za-z0-9/-]+)",
        r"\b(?:Receipt\s*No\.?|Receipt\s*Number)\s*[:\-]?\s*([A-Za-z0-9/-]+)",
        r"\b(?:Transaction\s*ID|Txn\s*ID)\s*[:\-]?\s*([A-Za-z0-9/-]+)",
    ]

    for pattern in patterns:
        matches = re.findall(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        for value in matches:
            value = value.strip()

            if value and value not in results:
                results.append(value)

    return results


# ============================================================
# MONETARY AMOUNTS
# ============================================================

def extract_amounts(text):
    results = []

    patterns = [
        r"(?:₹|Rs\.?|INR)\s*[\d,]+(?:\.\d{1,2})?",
        r"(?:amount|total|fee|fees|paid|payment)\s*(?:is|:|-)?\s*(?:₹|Rs\.?|INR)?\s*[\d,]+(?:\.\d{1,2})?",
    ]

    for pattern in patterns:
        matches = re.findall(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        for value in matches:
            value = clean_text(value)

            if value and value not in results:
                results.append(value)

    return results


# ============================================================
# GENERIC KEY-VALUE EXTRACTION
# ============================================================

FIELD_ALIASES = OrderedDict(
    [
        (
            "name",
            [
                "name",
                "student name",
                "candidate name",
                "applicant name",
                "holder name",
            ],
        ),
        (
            "registration_number",
            [
                "regd.no",
                "regd no",
                "registration no",
                "registration number",
                "register no",
            ],
        ),
        (
            "application_number",
            [
                "application no",
                "application number",
                "app no",
            ],
        ),
        (
            "reference_number",
            [
                "reference no",
                "reference number",
                "ref no",
            ],
        ),
        (
            "transaction_id",
            [
                "transaction id",
                "txn id",
                "transaction number",
            ],
        ),
        (
            "receipt_number",
            [
                "receipt no",
                "receipt number",
            ],
        ),
        (
            "date_of_birth",
            [
                "date of birth",
                "dob",
                "birth date",
            ],
        ),
        (
            "date",
            [
                "date",
            ],
        ),
        (
            "campus",
            [
                "campus",
            ],
        ),
        (
            "institute",
            [
                "institute",
                "institution",
            ],
        ),
        (
            "academic_year",
            [
                "academic year",
            ],
        ),
        (
            "percentage",
            [
                "percentage",
                "percent",
            ],
        ),
        (
            "disability_type",
            [
                "disability type",
                "type of disability",
            ],
        ),
        (
            "issuing_authority",
            [
                "issuing authority",
                "issued by",
            ],
        ),
        (
            "fee_description",
            [
                "fee description",
                "description",
            ],
        ),
        (
            "payment_mode",
            [
                "payment mode",
                "mode of payment",
            ],
        ),
        (
            "amount",
            [
                "amount",
                "total amount",
                "amount paid",
            ],
        ),
    ]
)


def extract_label_values(text):
    values = OrderedDict()

    for field, aliases in FIELD_ALIASES.items():

        for alias in aliases:

            pattern = (
                r"\b"
                + re.escape(alias)
                + r"\s*[:\-]\s*"
                r"(.+?)(?=\s+\b(?:"
                + "|".join(
                    re.escape(a)
                    for other_field, alias_list in FIELD_ALIASES.items()
                    if other_field != field
                    for a in alias_list
                )
                + r")\b\s*[:\-]|$)"
            )

            match = re.search(
                pattern,
                text,
                flags=re.IGNORECASE
            )

            if match:

                value = clean_text(
                    match.group(1)
                )

                if value:
                    values[field] = value
                    break

    return values


# ============================================================
# GITAM GRADE CARD DETECTION
# ============================================================

def is_grade_card(text):
    text_lower = text.lower()

    return (
        "grade card" in text_lower
        and (
            "sgpa" in text_lower
            or "cgpa" in text_lower
            or "course code" in text_lower
        )
    )


# ============================================================
# GITAM GRADE CARD FIELDS
# ============================================================

def extract_grade_card_fields(text):
    fields = OrderedDict()

    # --------------------------------------------------------
    # Degree
    # --------------------------------------------------------

    degree_match = re.search(
        r"\b(B\.?\s*Tech(?:\s+Degree)?|M\.?\s*Tech(?:\s+Degree)?|B\.?\s*Sc|M\.?\s*Sc|BCA|MCA)"
        r"(?:\s+Degree)?",
        text,
        flags=re.IGNORECASE
    )

    if degree_match:
        degree = clean_text(
            degree_match.group(0)
        )

        if "degree" not in degree.lower():
            degree = degree + " Degree"

        fields["course_degree"] = degree

    # --------------------------------------------------------
    # Registration Number
    # --------------------------------------------------------

    registration_match = re.search(
        r"\bRegd\.?\s*No\.?\s*[:\-]?\s*([A-Za-z0-9/-]+)",
        text,
        flags=re.IGNORECASE
    )

    if registration_match:
        fields["registration_number"] = (
            registration_match.group(1).strip()
        )

    # --------------------------------------------------------
    # Name
    # --------------------------------------------------------

    name_match = re.search(
        r"\bName\s*[:\-]\s*(.+?)(?=\s+Branch\s*[:\-])",
        text,
        flags=re.IGNORECASE
    )

    if name_match:
        fields["name"] = clean_text(
            name_match.group(1)
        )

    # --------------------------------------------------------
    # Branch
    # --------------------------------------------------------

    branch_match = re.search(
        r"\bBranch\s*[:\-]\s*(.+?)(?=\s+Course\s+Code\b)",
        text,
        flags=re.IGNORECASE
    )

    if branch_match:
        fields["branch"] = clean_text(
            branch_match.group(1)
        )

    # --------------------------------------------------------
    # Semester / Examination Period
    # --------------------------------------------------------

    semester_match = re.search(
        r"\b("
        r"(?:I|II|III|IV|V|VI|VII|VIII|1st|2nd|3rd|4th|5th|6th|7th|8th)"
        r"\s+Semester"
        r"(?:\s*,?\s*[A-Za-z]+\s+\d{4})?"
        r")",
        text,
        flags=re.IGNORECASE
    )

    if semester_match:
        semester_value = clean_text(
            semester_match.group(1)
        )

        fields["semester"] = semester_value
        fields["exam_period"] = semester_value

    # --------------------------------------------------------
    # Academic Year
    # --------------------------------------------------------

    academic_year_match = re.search(
        r"\b(20\d{2}\s*[-/]\s*20\d{2})\b",
        text
    )

    if academic_year_match:
        fields["academic_year"] = (
            academic_year_match.group(1)
        )

    # --------------------------------------------------------
    # SGPA
    # --------------------------------------------------------

    sgpa_match = re.search(
        r"\bSGPA\s*[:=]?\s*(\d+(?:\.\d+)?)",
        text,
        flags=re.IGNORECASE
    )

    if sgpa_match:
        fields["sgpa"] = sgpa_match.group(1)

    # --------------------------------------------------------
    # CGPA
    # --------------------------------------------------------

    cgpa_match = re.search(
        r"\bCGPA\s*[:=]?\s*(\d+(?:\.\d+)?)",
        text,
        flags=re.IGNORECASE
    )

    if cgpa_match:
        fields["cgpa"] = cgpa_match.group(1)

    # --------------------------------------------------------
    # Printed Date
    # --------------------------------------------------------

    printed_match = re.search(
        r"\bPrinted\s+On\s*[:\-]?\s*"
        r"(\d{1,2}[-/][A-Za-z]{3,9}[-/]\d{2,4})",
        text,
        flags=re.IGNORECASE
    )

    if printed_match:
        fields["printed_date"] = (
            printed_match.group(1)
        )

    # --------------------------------------------------------
    # Institution
    # --------------------------------------------------------

    if "GITAM" in text.upper():

        fields["institution"] = (
            "GITAM (Gandhi Institute of Technology and Management)"
        )

    return fields


# ============================================================
# GRADE CARD SUBJECT TABLE
# ============================================================

def extract_grade_card_table(text):
    """
    Extract Grade Card rows using the known structure:

        Course Code
        Course Name
        Credits
        Grade

    Example:

        24CSEN1001 Digital Logic Circuits 2 B+
    """

    rows = []

    # --------------------------------------------------------
    # Normalize OCR text
    # --------------------------------------------------------

    text = clean_text(text)

    # --------------------------------------------------------
    # Grade pattern
    # --------------------------------------------------------

    grade_pattern = (
        r"(?:O|A\+|A-|A|B\+|B-|B|C\+|C-|C|D\+|D-|D|E|F|P|S)"
    )

    # --------------------------------------------------------
    # Course-code pattern
    #
    # GITAM course codes commonly look like:
    # 24CSEN1001
    # 24CSEN1041
    # 24EECE2231
    # ENVS1003
    # IENT1051
    # LANG1251
    # MATH1272
    # PHYS1291
    # --------------------------------------------------------

    course_code_pattern = (
        r"\b(?:"
        r"\d{2}[A-Z]{2,6}\d{3,5}"
        r"|"
        r"[A-Z]{3,6}\d{3,5}"
        r")\b"
    )

    # --------------------------------------------------------
    # Find course-code positions
    # --------------------------------------------------------

    matches = list(
        re.finditer(
            course_code_pattern,
            text,
            flags=re.IGNORECASE
        )
    )

    if not matches:
        return rows

    # --------------------------------------------------------
    # Process each course row
    # --------------------------------------------------------

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

        # ----------------------------------------------------
        # Stop before SGPA / CGPA / Printed On / Note
        # ----------------------------------------------------

        segment = re.split(
            r"\b(?:SGPA|CGPA|Printed\s+On|Note)\b",
            segment,
            maxsplit=1,
            flags=re.IGNORECASE
        )[0]

        segment = clean_text(segment)

        if not segment:
            continue

        # ----------------------------------------------------
        # Extract credits + grade from the END of the segment
        #
        # Examples:
        #
        # Digital Logic Circuits 2 B+
        # Environmental Studies 3 A
        # Fundamentals of Entrepreneurship 2 A+
        # ----------------------------------------------------

        tail_pattern = (
            r"^(.*?)"
            r"\s+"
            r"(\d+(?:\.\d+)?)"
            r"\s+"
            r"("
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

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        if not course_name:
            continue

        if len(course_name) < 2:
            continue

        if len(course_name) > 150:
            continue

        # Avoid header/metadata false positives.
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

    # --------------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------------

    unique_rows = []
    seen = set()

    for row in rows:

        key = tuple(row)

        if key not in seen:
            seen.add(key)
            unique_rows.append(row)

    return unique_rows


# ============================================================
# RECEIPT EXTRACTION
# ============================================================

def extract_receipt_fields(text):
    fields = OrderedDict()

    patterns = OrderedDict(
        [
            (
                "transaction_id",
                r"\b(?:Transaction\s*ID|Txn\s*ID)\s*[:\-]?\s*([A-Za-z0-9/-]+)"
            ),
            (
                "receipt_number",
                r"\b(?:Receipt\s*No\.?|Receipt\s*Number)\s*[:\-]?\s*([A-Za-z0-9/-]+)"
            ),
            (
                "registration_number",
                r"\b(?:Regd\.?\s*No\.?|Registration\s*No\.?)\s*[:\-]?\s*([A-Za-z0-9/-]+)"
            ),
            (
                "payment_mode",
                r"\b(?:Payment\s*Mode|Mode\s*of\s*Payment)\s*[:\-]?\s*(.+?)(?=\s+\b(?:Amount|Total|Date|Transaction|Txn)\b|$)"
            ),
            (
                "fee_description",
                r"\b(?:Fee\s*Description|Description)\s*[:\-]?\s*(.+?)(?=\s+\b(?:Amount|Total|Payment|Date)\b|$)"
            ),
        ]
    )

    for field, pattern in patterns.items():

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        if match:

            value = clean_text(
                match.group(1)
            )

            if value:
                fields[field] = value

    amount_match = re.search(
        r"\b(?:Total\s+Amount|Amount\s+Paid|Amount|Total)\s*"
        r"[:\-]?\s*(?:₹|Rs\.?|INR)?\s*"
        r"([\d,]+(?:\.\d{1,2})?)",
        text,
        flags=re.IGNORECASE
    )

    if amount_match:
        fields["amount"] = amount_match.group(1)

    return fields


# ============================================================
# CERTIFICATE EXTRACTION
# ============================================================

def extract_certificate_fields(text):
    fields = OrderedDict()

    patterns = [
        (
            "application_number",
            r"\b(?:Application\s*No\.?|Application\s*Number)\s*[:\-]?\s*([A-Za-z0-9/-]+)"
        ),
        (
            "reference_number",
            r"\b(?:Reference\s*No\.?|Reference\s*Number)\s*[:\-]?\s*([A-Za-z0-9/-]+)"
        ),
        (
            "name",
            r"\bName\s*[:\-]\s*(.+?)(?=\s+\b(?:Date|DOB|Father|Mother|Address|Gender)\b|$)"
        ),
        (
            "date_of_birth",
            r"\b(?:Date\s+of\s+Birth|DOB)\s*[:\-]?\s*([A-Za-z0-9/-]+)"
        ),
        (
            "disability_type",
            r"\b(?:Disability\s*Type|Type\s+of\s+Disability)\s*[:\-]?\s*(.+?)(?=\s+\b(?:Percentage|Date|ID)\b|$)"
        ),
        (
            "issuing_authority",
            r"\b(?:Issuing\s+Authority|Issued\s+By)\s*[:\-]?\s*(.+?)(?=\s+\b(?:Date|ID|Application)\b|$)"
        ),
    ]

    for field, pattern in patterns:

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        if match:

            value = clean_text(
                match.group(1)
            )

            if value:
                fields[field] = value

    percentage_match = re.search(
        r"\b(\d{1,3}(?:\.\d+)?)\s*%",
        text
    )

    if percentage_match:
        fields["percentage"] = (
            percentage_match.group(1) + "%"
        )

    return fields


# ============================================================
# ACADEMIC INFORMATION
# ============================================================

def extract_academic_information(text):
    information = OrderedDict()

    if "GITAM" in text.upper():

        information["institution"] = (
            "GITAM (Gandhi Institute of Technology and Management)"
        )

    else:

        institution_patterns = [
            r"^(.+?)(?=\s+GRADE\s+CARD\b)",
            r"^(.+?)(?=\s+MARKS?\s+MEMO\b)",
            r"^(.+?)(?=\s+TRANSCRIPT\b)",
        ]

        for pattern in institution_patterns:

            match = re.search(
                pattern,
                text,
                flags=re.IGNORECASE
            )

            if match:

                institution = clean_text(
                    match.group(1)
                )

                if institution:
                    information["institution"] = institution
                    break

    academic_year_match = re.search(
        r"\b(20\d{2}\s*[-/]\s*20\d{2})\b",
        text
    )

    if academic_year_match:
        information["academic_year"] = (
            academic_year_match.group(1)
        )

    result_patterns = [
        r"\b(First\s+Class)\b",
        r"\b(Second\s+Class)\b",
        r"\b(Distinction)\b",
        r"\b(Pass)\b",
    ]

    for pattern in result_patterns:

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        if match:

            information["result"] = clean_text(
                match.group(1)
            )

            break

    return information


# ============================================================
# GENERIC ACADEMIC FIELDS
# ============================================================

def extract_academic_fields(text):
    fields = OrderedDict()

    if is_grade_card(text):

        grade_card_fields = extract_grade_card_fields(
            text
        )

        for key, value in grade_card_fields.items():
            fields[key] = value

        return fields

    registration_match = re.search(
        r"\b(?:Regd\.?\s*No\.?|Registration\s*No\.?|Registration\s*Number)\s*[:\-]?\s*([A-Za-z0-9/-]+)",
        text,
        flags=re.IGNORECASE
    )

    if registration_match:
        fields["registration_number"] = (
            registration_match.group(1)
        )

    name_match = re.search(
        r"\bName\s*[:\-]\s*(.+?)(?=\s+\b(?:Branch|Course|Registration|Regd)\b|$)",
        text,
        flags=re.IGNORECASE
    )

    if name_match:
        fields["name"] = clean_text(
            name_match.group(1)
        )

    degree_match = re.search(
        r"\b(B\.?\s*Tech|M\.?\s*Tech|BCA|MCA|B\.?\s*Sc|M\.?\s*Sc)\b",
        text,
        flags=re.IGNORECASE
    )

    if degree_match:
        fields["course_degree"] = clean_text(
            degree_match.group(1)
        )

    cgpa_match = re.search(
        r"\bCGPA\s*[:=]?\s*(\d+(?:\.\d+)?)",
        text,
        flags=re.IGNORECASE
    )

    if cgpa_match:
        fields["cgpa"] = cgpa_match.group(1)

    sgpa_match = re.search(
        r"\bSGPA\s*[:=]?\s*(\d+(?:\.\d+)?)",
        text,
        flags=re.IGNORECASE
    )

    if sgpa_match:
        fields["sgpa"] = sgpa_match.group(1)

    percentage_match = re.search(
        r"\b(?:Percentage|Percent)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*%?",
        text,
        flags=re.IGNORECASE
    )

    if percentage_match:
        fields["percentage"] = (
            percentage_match.group(1) + "%"
        )

    return fields


# ============================================================
# TABLE-LIKE DATA
# ============================================================

def extract_table_data(text, document_type):
    if document_type == "Academic Document" and is_grade_card(text):

        rows = extract_grade_card_table(text)

        return {
            "headers": [
                "Course Code",
                "Name of the Course",
                "Credits",
                "Grade",
            ],
            "rows": rows,
        }

    return {
        "headers": [],
        "rows": [],
    }


# ============================================================
# NUMBERED ENTRIES
# ============================================================

def extract_numbered_entries(text):
    entries = []

    matches = re.findall(
        r"(?:^|\s)(\d+)\.\s*([^0-9]{10,})",
        text
    )

    for number, value in matches:

        value = clean_text(value)

        if not value:
            continue

        if len(value) > 300:
            value = value[:300]

        entries.append(
            {
                "number": number,
                "text": value,
            }
        )

    return entries


# ============================================================
# GENERIC ENTITIES
# ============================================================

def extract_generic_entities(text, document_type):
    entities = OrderedDict()

    entities["dates"] = extract_dates(text)
    entities["emails"] = extract_emails(text)
    entities["phone_numbers"] = extract_phone_numbers(text)
    entities["urls"] = extract_urls(text)
    entities["reference_numbers"] = extract_reference_numbers(text)

    if document_type == "Receipt / Invoice":
        entities["amounts"] = extract_amounts(text)
    else:
        entities["amounts"] = []

    entities["label_values"] = extract_label_values(
        text
    )

    return entities


# ============================================================
# MERGE FIELD VALUES
# ============================================================

def merge_fields(target, source):
    for key, value in source.items():

        if value is None:
            continue

        if isinstance(value, str):
            value = clean_text(value)

        if value:
            target[key] = value


# ============================================================
# MAIN DOCUMENT INFORMATION EXTRACTION
# ============================================================

def extract_document_info(words, boxes=None):

    if words is None:
        words = []

    if not isinstance(words, list):
        words = list(words)

    text = build_combined_text(
        words
    )

    document_type = detect_document_type(
        text
    )

    entities = extract_generic_entities(
        text,
        document_type
    )

    structured_fields = OrderedDict()

    merge_fields(
        structured_fields,
        entities.get(
            "label_values",
            {}
        )
    )

    if document_type == "Academic Document":

        academic_fields = extract_academic_fields(
            text
        )

        merge_fields(
            structured_fields,
            academic_fields
        )

        academic_information = (
            extract_academic_information(
                text
            )
        )

        if is_grade_card(text):

            grade_card_fields = (
                extract_grade_card_fields(
                    text
                )
            )

            merge_fields(
                structured_fields,
                grade_card_fields
            )

        entities["academic_information"] = (
            academic_information
        )

    elif document_type == "Receipt / Invoice":

        receipt_fields = extract_receipt_fields(
            text
        )

        merge_fields(
            structured_fields,
            receipt_fields
        )

    elif document_type == "Certificate":

        certificate_fields = (
            extract_certificate_fields(
                text
            )
        )

        merge_fields(
            structured_fields,
            certificate_fields
        )

    else:

        entities["academic_information"] = (
            extract_academic_information(
                text
            )
        )

    # --------------------------------------------------------
    # Grade Card table
    # --------------------------------------------------------

    table_data = extract_table_data(
        text,
        document_type
    )

    entities["table_headers"] = (
        table_data["headers"]
    )

    entities["table_rows"] = (
        table_data["rows"]
    )

    # --------------------------------------------------------
    # Numbered entries
    # --------------------------------------------------------

    if document_type == "Academic Document":
        entities["numbered_entries"] = []
    else:
        entities["numbered_entries"] = (
            extract_numbered_entries(
                text
            )
        )

    # --------------------------------------------------------
    # Final structured fields
    # --------------------------------------------------------

    entities["label_values"] = (
        structured_fields
    )

    document_information = OrderedDict()

    document_information["document_type"] = (
        document_type
    )

    document_information["text"] = text

    document_information["entities"] = entities

    return {
        "document_information": document_information,
        "document_type": document_type,
        "text": text,
        "entities": entities,
    }