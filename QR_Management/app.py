import streamlit as st
import pandas as pd
import plotly as px
from Scan_QR import scan

FILE = "QR_Asset_Dataset.xlsx"  # FIXED: matches the actual generated filename

# Load Excel data
assets = pd.read_excel(FILE, sheet_name="Assets")
employees = pd.read_excel(FILE, sheet_name="Employees")
vendors = pd.read_excel(FILE, sheet_name="Vendors")
maintenance = pd.read_excel(FILE, sheet_name="Maintenance")

st.set_page_config(page_title="QR Asset Management", page_icon="📦")
st.title("📦 QR Asset Management System")

# Dashboard
st.subheader("📊 Asset Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Assets", len(assets))
col2.metric("Purchase Value", f"₹{assets['Purchase_Cost'].sum():,.0f}")
col3.metric("Current Value", f"₹{assets['Current_Value'].sum():,.0f}")
col4.metric("Maintenance Cost", f"₹{maintenance['Maintenance_Cost'].sum():,.0f}")

st.divider()

# Asset status and category charts
col1, col2 = st.columns(2)
with col1:
    status = assets["Status"].value_counts().reset_index()
    status.columns = ["Status", "Count"]
    fig = px.pie(status, names="Status", values="Count", title="Asset Status")
    st.plotly_chart(fig, use_container_width=True)
with col2:
    category = assets["Category"].value_counts().reset_index()
    category.columns = ["Category", "Count"]
    fig = px.bar(category, x="Category", y="Count", title="Assets By Category")
    st.plotly_chart(fig, use_container_width=True)

# Employee asset allocation
st.subheader("👥 Employee Asset Allocation")
employee_assets = assets.groupby("Employee_ID").size().reset_index(name="Asset_Count")
employee_assets = employee_assets.merge(employees, on="Employee_ID", how="left")
st.dataframe(employee_assets[["Employee_Name", "Department", "Asset_Count"]], use_container_width=True)

st.divider()

# QR Scanner
if st.button("📷 Scan Asset QR"):
    scanned_id = scan()
    if scanned_id:
        st.session_state["asset_id"] = scanned_id
    else:
        st.error("QR not detected")

asset_id = st.session_state.get("asset_id")

if asset_id:
    st.success(f"Scanned Asset ID : {asset_id}")
    asset_data = assets[assets["Asset_ID"] == asset_id]

    if not asset_data.empty:
        asset = asset_data.iloc[0]

        # Asset details
        st.header("📦 Asset Information")
        st.write(f"""
        **Asset ID:** {asset['Asset_ID']}        
        **Asset Name:** {asset['Asset_Name']}        
        **Category:** {asset['Category']}        
        **Status:** {asset['Status']}        
        **Purchase Cost:** ₹{asset['Purchase_Cost']}        
        **Current Value:** ₹{asset['Current_Value']}        
        **Employee ID:** {asset['Employee_ID']}        
        **Vendor ID:** {asset['Vendor_ID']}
        """)

        # Employee details
        st.header("👤 Assigned Employee")
        employee_data = employees[employees["Employee_ID"] == asset["Employee_ID"]]
        if not employee_data.empty:
            employee = employee_data.iloc[0]
            st.write(f"""
            **Employee ID:** {employee['Employee_ID']}            
            **Name:** {employee['Employee_Name']}           
            **Department:** {employee['Department']}            
            **Email:** {employee['Email']}            
            **Phone:** {employee['Phone_Number']}
            """)
        else:
            st.warning("Employee details not found")

        # Vendor details
        st.header("🏢 Vendor Details")
        vendor_data = vendors[vendors["Vendor_ID"] == asset["Vendor_ID"]]
        if not vendor_data.empty:
            vendor = vendor_data.iloc[0]
            st.write(f"""
            **Vendor ID:** {vendor['Vendor_ID']}            
            **Vendor Name:** {vendor['Vendor_Name']}            
            **Contact:** {vendor['Contact_Number']}            
            **Email:** {vendor['Email']}
            """)
        else:
            st.warning("Vendor details not found")

        # Maintenance details
        st.header("🛠 Maintenance History")
        history = maintenance[maintenance["Asset_ID"] == asset_id]
        if history.empty:
            st.info("No maintenance records available")
        else:
            for index, row in history.iterrows():
                st.markdown("---")
                st.write(f"""
                📅 **Date:** {row['Maintenance_Date']}                
                🔧 **Service:** {row['Issue']}                
                💰 **Cost:** ₹{row['Maintenance_Cost']}                
                📝 **Status:** {row['Status']}
                """)
    else:
        st.error("Asset not found")
else:
    st.info("Click Scan Asset QR to view asset details")
