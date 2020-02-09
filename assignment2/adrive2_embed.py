###!!!! This version works when embedded in a template file, see assignment2/template.py !!!!
### For separated use, see adrive2_new.py
import sys,traceback
def a2answers(gdict,errlog):
  globals().update(gdict)
  errs=0
  (cerrs,ans)=carefulBind(
    [('a1a','sorted(model.states)'),
     ('a1b','len(model.emission_PD["VERB"].samples()) if type(model.emission_PD)==nltk.probability.ConditionalProbDist else FAILED'),
     ('a1c','-model.elprob("VERB","attack")'),
     ('a1d','model.emission_PD._probdist_factory.__class__.__name__ if model.emission_PD is not None else FAILED'),
     ('a2a','len(model.transition_PD["VERB"].samples()) if type(model.transition_PD)==nltk.probability.ConditionalProbDist else FAILED'),
     ('a2b','-model.tlprob("VERB","DET")'),
     #('a3a','(vqq,vtt)'), Can't figure out how to do these two...
     #('a3b','(bqq,btt)'), Were size of state and time dimensions of viterbi and backpointer reprs
     ('a4a3','accuracy'),
     ('a4b1','bad_tags'),
     ('a4b2','good_tags'),
     ('a4b3','answer4b'),
     ('a4c','model.get_viterbi_value("VERB",5)'),
     ('a4d','min((model.get_viterbi_value(s,-1) for s in model.states)) if len(model.states)>0 else FAILED'),
     ('a4e','list(ttags)'),
     ('a5','answer5'),
     ('a6','answer6')
     ],globals(),errlog)
  errs+=cerrs
  try:
    model.initialise('attack')
  except NotImplementedError:
    pass
  except Exception as e:
    errs+=1
    print("Exception in initialising model in adrive2:\n%s"%repr(e),
          file=errlog)
    traceback.print_tb(sys.exc_info()[2],None,errlog)
  (cerrs,nans)=carefulBind(
    [('a3c','model.get_viterbi_value("VERB",0)'),
     ('a3d','model.get_backpointer_value("VERB",0)')],globals(),errlog)
  ans.update(nans)
  errs+=cerrs
  return (ans,errs)

if __name__ == '__main__':
  from autodrive_embed import run, answers, HMM, carefulBind
  with open("userErrs.txt","w") as errlog:
    run(answers,a2answers,errlog)
