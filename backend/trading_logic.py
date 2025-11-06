"""
Energy Trading Dashboard - Core Logic
"""

def simulate_energy_trade(demand, budget, energy_in_stock, price_per_mw):
    result = {
        "demand": demand,
        "available_energy": energy_in_stock,
        "budget": budget,
        "price_per_mw": price_per_mw,
    }

    # Case 1: Energy shortage
    if energy_in_stock < demand:
        energy_sold = energy_in_stock
        revenue = energy_sold * price_per_mw
        demand_met_percent = (energy_sold / demand) * 100
        result.update({
            "status": "shortage",
            "energy_sold": energy_sold,
            "revenue": revenue,
            "remaining_energy": 0,
            "demand_met_percent": demand_met_percent,
            "message": f"⚠️ Energy shortage! Only {demand_met_percent:.1f}% of demand met.",
            "icon": "⚠️",
            "color": "red"
        })
        return result

    # Case 2: Exact match
    if energy_in_stock == demand:
        required_payment = demand * price_per_mw
        if budget >= required_payment:
            result.update({
                "status": "fulfilled",
                "energy_sold": demand,
                "revenue": required_payment,
                "remaining_energy": 0,
                "demand_met_percent": 100.0,
                "message": "✅ Full demand met. All energy sold.",
                "icon": "✅",
                "color": "green"
            })
        else:
            energy_sold = budget / price_per_mw
            revenue = budget
            result.update({
                "status": "partial_budget",
                "energy_sold": energy_sold,
                "revenue": revenue,
                "remaining_energy": energy_in_stock - energy_sold,
                "demand_met_percent": (energy_sold / demand) * 100,
                "message": "💰 Partial supply due to budget limit.",
                "icon": "💰",
                "color": "orange"
            })
        return result

    # Case 3: Surplus energy
    if energy_in_stock > demand:
        required_payment = demand * price_per_mw
        if budget >= required_payment:
            result.update({
                "status": "fulfilled",
                "energy_sold": demand,
                "revenue": required_payment,
                "remaining_energy": energy_in_stock - demand,
                "demand_met_percent": 100.0,
                "message": "✅ Full demand met. Surplus available.",
                "icon": "✅",
                "color": "green"
            })
        else:
            energy_sold = budget / price_per_mw
            revenue = budget
            demand_met_percent = (energy_sold / demand) * 100
            result.update({
                "status": "partial_budget",
                "energy_sold": energy_sold,
                "revenue": revenue,
                "remaining_energy": energy_in_stock - energy_sold,
                "demand_met_percent": demand_met_percent,
                "message": f"💰 Partial supply based on budget ({demand_met_percent:.1f}% met).",
                "icon": "💰",
                "color": "orange"
            })
        return result

