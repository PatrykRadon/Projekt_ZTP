# import pandas as pd
import pandas as pd
from uuid import uuid4
from datetime import datetime, timedelta

# # create a sample DataFrame
# df = pd.DataFrame({'sqare_meters': [1, 2, 3], 'rooms': [1, 2, 3],
#                    'age': [1, 2, 3], 'price': [1, 2, 3],
#                    'expiration_timestamp': [datetime.now() + timedelta(days=2137) for i in range(3)],
#                    'id': [str(uuid4()) for _ in range(3)]
#                    })
#
# df.to_parquet('./data/active_set.parquet')

df_test = pd.read_parquet('./data/active_set.parquet')

for index, row in df_test.iterrows():
    print(row)