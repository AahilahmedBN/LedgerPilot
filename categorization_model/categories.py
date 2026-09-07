"""
LedgerPilot — Category & ITC eligibility definitions.

"""

from dataclasses import dataclass, field


@dataclass
class Category:
    key: str
    label: str
    gst_rate: str  # "0" | "5" | "12" | "18" | "28" | "exempt"
    itc_eligible: str  # "Eligible" | "Blocked" | "Partially Eligible" | "Not Applicable"
    itc_reason: str
    document_types: list = field(default_factory=lambda: ["invoice", "receipt", "bank_statement_line"])
    # Template phrases used to generate description_raw / description_clean.
    # {vendor} and {item} are filled in by the generator.
    templates: list = field(default_factory=list)
    items: list = field(default_factory=list)


CATEGORIES = [
    Category(
        key="purchase_goods",
        label="Purchase of Goods (Raw Material/Trading Stock)",
        gst_rate="18",
        itc_eligible="Eligible",
        itc_reason="Used in the course/furtherance of business — ordinary input goods, no Sec 17(5) restriction applies.",
        templates=["Purchase of {item} from {vendor}", "{vendor} - {item} supply", "Payment to {vendor} for {item}"],
        items=["raw cotton", "packaging material", "steel sheets", "plastic granules", "printed labels", "raw plastic pellets", "yarn", "adhesive tape rolls"],
    ),
    Category(
        key="capital_goods",
        label="Capital Goods",
        gst_rate="18",
        itc_eligible="Eligible",
        itc_reason="Capital goods used for business are ITC eligible under Sec 16, subject to depreciation not being claimed on the tax component (Sec 16(3)).",
        templates=["Purchase of {item} - {vendor}", "{vendor} - Capital Asset Purchase - {item}", "{item} installation invoice from {vendor}"],
        items=["industrial sewing machine", "CNC lathe", "office desktop computers", "commercial printer", "generator set", "packing machine", "3-phase motor"],
    ),
    Category(
        key="input_services",
        label="Input Services (Professional/Consulting/IT)",
        gst_rate="18",
        itc_eligible="Eligible",
        itc_reason="Professional/consulting/IT services used for business operations are standard eligible input services.",
        templates=["{vendor} - {item} services invoice", "Professional fees to {vendor} for {item}", "{item} charges - {vendor}"],
        items=["chartered accountant", "legal consultation", "IT support", "software subscription", "website maintenance", "auditing services", "payroll processing"],
    ),
    Category(
        key="rent",
        label="Rent/Lease of Business Premises",
        gst_rate="18",
        itc_eligible="Eligible",
        itc_reason="Rent on commercial premises used for business is an eligible input service.",
        templates=["Monthly rent - {vendor}", "{vendor} - Shop Rent Payment", "Lease rental to {vendor} for {item}"],
        items=["shop premises", "warehouse space", "office unit", "godown"],
    ),
    Category(
        key="utilities",
        label="Utilities (Business Electricity/Internet)",
        gst_rate="18",
        itc_eligible="Eligible",
        itc_reason="Utilities consumed for business premises are eligible input services (note: electricity itself is often outside GST, but the connection/service charges from a GST-registered provider may attract GST).",
        templates=["{vendor} - {item} bill payment", "{item} charges - {vendor}"],
        items=["electricity", "broadband internet", "water supply", "telephone/mobile plan"],
    ),
    Category(
        key="bank_charges",
        label="Bank Charges/Financial Services",
        gst_rate="18",
        itc_eligible="Eligible",
        itc_reason="Bank service charges (not interest) are taxable supplies of financial services and are ordinarily ITC eligible input services.",
        templates=["{vendor} - Bank Charges", "{item} fee - {vendor}", "{vendor} - {item}"],
        items=["processing fee", "account maintenance charge", "cheque bounce fee", "loan processing fee", "POS machine rental"],
    ),
    Category(
        key="motor_vehicle",
        label="Motor Vehicle Expenses",
        gst_rate="28",
        itc_eligible="Blocked",
        itc_reason="Sec 17(5)(a) blocks ITC on motor vehicles for transport of persons (seating <=13) and related services, unless used for further supply, passenger transport, or driving training.",
        templates=["{vendor} - {item}", "Vehicle {item} - {vendor}", "Payment to {vendor} for {item}"],
        items=["car servicing", "vehicle insurance premium", "fuel purchase", "car repair", "vehicle registration fee"],
    ),
    Category(
        key="food_catering",
        label="Food & Beverages/Outdoor Catering",
        gst_rate="5",
        itc_eligible="Blocked",
        itc_reason="Sec 17(5)(b)(i) explicitly blocks ITC on food and beverages and outdoor catering, unless used to make an outward taxable supply of the same category or as part of a composite/mixed supply.",
        templates=["{vendor} - {item}", "Office {item} order - {vendor}", "Catering payment - {vendor}"],
        items=["staff lunch catering", "client meeting refreshments", "office tea/coffee supplies", "festival sweets order"],
    ),
    Category(
        key="club_membership",
        label="Club/Health/Fitness Memberships",
        gst_rate="18",
        itc_eligible="Blocked",
        itc_reason="Sec 17(5)(b)(ii) blocks ITC on membership of a club, health, or fitness centre.",
        templates=["{vendor} - Membership Fee", "{item} - {vendor}"],
        items=["gym membership", "club membership renewal", "wellness center package"],
    ),
    Category(
        key="employee_travel",
        label="Employee Travel Benefits",
        gst_rate="5",
        itc_eligible="Blocked",
        itc_reason="Sec 17(5)(b)(iii) blocks ITC on travel benefits extended to employees (e.g. leave/home travel concession), unless obligatory under law.",
        templates=["{item} - {vendor}", "Employee {item} booking - {vendor}"],
        items=["home travel concession booking", "employee leave travel airfare", "staff outing travel booking"],
    ),
    Category(
        key="employee_insurance",
        label="Employee Insurance/Reimbursements",
        gst_rate="18",
        itc_eligible="Blocked",
        itc_reason="Sec 17(5)(b) blocks ITC on health/life insurance for employees unless it is obligatory for the employer under any law for the time being in force.",
        templates=["{vendor} - {item}", "{item} premium payment - {vendor}"],
        items=["employee group health insurance", "staff life insurance premium", "employee medical reimbursement"],
    ),
    Category(
        key="works_contract",
        label="Works Contract/Construction of Immovable Property",
        gst_rate="18",
        itc_eligible="Blocked",
        itc_reason="Sec 17(5)(c)/(d) blocks ITC on works contract services and goods/services for construction of immovable property (other than plant & machinery), even when used for business.",
        templates=["{vendor} - {item}", "Construction payment - {vendor} for {item}"],
        items=["shop renovation works", "warehouse construction contract", "building repair works contract", "flooring and fixtures work"],
    ),
    Category(
        key="sales_revenue",
        label="Sales/Outward Supply",
        gst_rate="18",
        itc_eligible="Not Applicable",
        itc_reason="This is outward supply (revenue), not a purchase — it is relevant to output tax liability, not input tax credit.",
        templates=["Sale invoice to {vendor}", "{vendor} - Sales Invoice", "Payment received from {vendor} for {item}"],
        items=["finished goods", "bulk order", "retail sale", "export order", "wholesale supply"],
    ),
    Category(
        key="exempt_non_gst",
        label="Exempt/Non-GST Expense",
        gst_rate="exempt",
        itc_eligible="Not Applicable",
        itc_reason="Salaries, government fees, and similar payments are outside the scope of GST entirely — no ITC question arises.",
        templates=["{item} payment", "{vendor} - {item}"],
        items=["staff salary payment", "government license fee", "municipal tax payment", "employee PF contribution", "professional tax payment"],
    ),
]

CATEGORY_BY_KEY = {c.key: c for c in CATEGORIES}

assert len(CATEGORIES) == 14, f"Expected 14 categories, got {len(CATEGORIES)}"
