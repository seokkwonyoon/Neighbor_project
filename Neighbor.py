import cvxpy as cp
import numpy as np
import pandas as pd
import json
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
location_group = None
list_of_location = None
app = FastAPI(title="Neighbor Local API")

@app.on_event("startup")
def load_data():
    """서비스 시작 시 listings.json을 읽어 전역 상태 초기화"""
    global location_group, list_of_location
    df = pd.read_json(BASE_DIR / "listings.json")
    df.set_index("id", inplace=True)
    location_group = df.groupby("location_id")
    list_of_location = list(location_group.groups)
    print("[startup] listings loaded:", len(df), "rows,", len(list_of_location), "locations")






@app.post("/")
def solve(payload: List[Dict[str, Any]]):
    
    try:
        Input = cars_info(payload, width=10, as_list=False) 
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid payload: {e}")


    results: List[Dict[str, Any]] = []
    for loc_id in list_of_location:
        sub_location = location_group.get_group(loc_id)
        pairs = sub_location[['length', 'width']].to_numpy()
        cost  = sub_location['price_in_cents'].to_numpy()

        res = optimal_cost(Input, pairs, cost) 
        if res is None:
            continue

        optimal_price, bin_used, _ = res
        idx_array = sub_location.index.to_numpy()
        selected_listing_ids = idx_array[bin_used].tolist()
        price_int = int(round(float(optimal_price)))  

        results.append(return_format(loc_id, selected_listing_ids, price_int))

    results.sort(key=lambda r: float(r["total_price_in_cents"]))
    return results




# Optimal function
def optimal_cost(items, bins, costs, solver="ECOS_BB", verbose=False):
   
    items = np.asarray(items, dtype=float)
    bins  = np.asarray(bins,  dtype=float)
    costs = np.asarray(costs, dtype=float)

    n_items = items.shape[0]
    n_bins  = bins.shape[0]


    feasible_mask = ((items[:, None, 0] <= bins[None, :, 0]) &
                     (items[:, None, 1] <= bins[None, :, 1]))  # (n_items, n_bins)
    if not feasible_mask.any(axis=1).all():
        return None  

  
    x = cp.Variable((n_items, n_bins), boolean=True)  
    y = cp.Variable(n_bins, boolean=True)           

    
    constraints = []
    constraints.append(cp.sum(x, axis=1) == 1)            
    constraints.append(x <= feasible_mask.astype(float))     

 
    load = items.T @ x                                       
    cap  = cp.multiply(bins.T, cp.vstack([y, y]))          
    constraints.append(load <= cap)


    objective = cp.Minimize(costs @ y)


    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=solver, verbose=verbose)  
    except Exception as e:
        print("[DEBUG] cvxpy solve failed:", repr(e))
        raise 

    if prob.status not in ("optimal", "optimal_inaccurate"):
        return None  

  
    x_val = x.value
    y_val = y.value

    used_bins = [int(j) for j in range(n_bins) if y_val[j] > 0.5]
    assignments = {j: np.where(x_val[:, j] > 0.5)[0].astype(int).tolist() for j in used_bins}

    return prob.value, used_bins,assignments


# Get cars info function
def cars_info(payload, width=10, as_list=False):
    data = json.loads(payload) if isinstance(payload, str) else list(payload)
    df = pd.DataFrame(data)

    if 'quantity' not in df:
        df['quantity'] = 1
    df['quantity'] = df['quantity'].fillna(1).astype(int)
    df['length']   = df['length'].astype(int)

    lengths = df['length'].repeat(df['quantity']).to_numpy()
    out = np.column_stack((lengths, np.full(lengths.size, width, dtype=int))) 

    return out.tolist() if as_list else out


# Change return format function
def return_format(location_id, listing_ids, total_price_in_cents):
    return {
        "location_id": location_id,
        "listing_ids": listing_ids if isinstance(listing_ids, list) else [listing_ids],
        "total_price_in_cents": total_price_in_cents,
    }



if __name__ == "__main__":
    data = pd.read_json("listings.json")
    data.set_index("id", inplace=True)

    location_group = data.groupby("location_id")
    list_of_location = list(location_group.groups)

    The_list_result = []
    for i in range(len(list_of_location)):
        sub_location = location_group.get_group(list_of_location[i])
        pairs = sub_location[['length', 'width']].to_numpy()
        cost  = sub_location['price_in_cents'].to_numpy()

        Input = np.array([[10, 10], [25, 10], [15, 10], [15, 10]])

        res = optimal_cost(Input, pairs, cost)
        if res is None:
            continue

        optimal_price, bin_used, assinged = res
        loc_id = list_of_location[i]
        idx_array = sub_location.index.to_numpy()
        selected_listing_ids = idx_array[bin_used].tolist()
        prices = float(optimal_price)
        The_list_result.append(return_format(loc_id, selected_listing_ids, prices))

 
    The_list_result.sort(key=lambda x: float(x["total_price_in_cents"]))

    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)






