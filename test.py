import json

req_json ='''{
    "imageTypes": ["RSRP热力图", "SINR热力图"],
    "category": "baseStation",
    "subOption": 1,
    "dataSource": "viavi",
    "downloadParams": {
        "mainCategory": "urban_coverage",
        "features": ["信号质量测量", "吞吐"],
        "startTime": "1739758277000",
        "endTime": "1739758877000"
    }
}'''


beq_json='''{imageTypes:[RSRP热力图,SINR热力图],category:baseStation,subOption:1,dataSource:viavi,downloadParams:{mainCategory:urban_coverage,features:[信号质量测量,吞吐],startTime:1739758277000,endTime:1739758877000}}'''


# {"imageTypes":["RSRP热力图","SINR热力图"],"category":"baseStation","subOption":1,"dataSource":"viavi","downloadParams":{"mainCategory":"urban_coverage","features":["信号质量测量","吞吐"],"startTime":"1739758277000","endTime":"1739758877000"}}

# {imageTypes:[RSRP热力图,SINR热力图],category:baseStation,subOption:1,dataSource:viavi,downloadParams:{mainCategory:urban_coverage,features:[信号质量测量,吞吐],startTime:1739758277000,endTime:1739758877000}}

print(f"Received test JSON string: {req_json}")

print(f"Received test JSON string: {beq_json}")

# req = json.loads(req_json)

beq = json.loads(beq_json)