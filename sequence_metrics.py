"""Length bookkeeping and speaker-holdout reporting helpers (no TensorFlow required)."""
def batch_lengths(frame_count,batch_size=75):
 if type(frame_count) is not int or frame_count<=0 or type(batch_size) is not int or batch_size<=0:raise ValueError("positive frame and batch counts required")
 full,remainder=divmod(frame_count,batch_size)
 return [batch_size]*full+([remainder] if remainder else [])

def edit_distance(reference,prediction):
 previous=list(range(len(prediction)+1))
 for i,a in enumerate(reference,1):
  row=[i]
  for j,b in enumerate(prediction,1):row.append(min(row[-1]+1,previous[j]+1,previous[j-1]+(a!=b)))
  previous=row
 return previous[-1]

def error_rates(reference,prediction):
 if not reference:raise ValueError("nonempty reference required")
 words=reference.split()
 return {"cer":edit_distance(reference,prediction)/len(reference),"wer":edit_distance(words,prediction.split())/len(words) if words else None}
