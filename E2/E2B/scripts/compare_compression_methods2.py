
#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import bz2
from scipy.optimize import curve_fit
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

METHOD_COLORS={'fine_tuning':'#1f77b4','basic_pruning':'#ff7f0e','advanced_pruning':'#2ca02c','quantization':'#d62728','magnitude':'#9467bd','structured':'#8c564b','block_sparse':'#e377c2','sparsegpt':'#7f7f7f'}

class CompressionAnalyzer:
    def __init__(self,results_dirs:List[str],output_dir:str):
        self.results_dirs=results_dirs
        self.output_dir=Path(output_dir);self.output_dir.mkdir(parents=True,exist_ok=True)
        self.combined_df=None
        self.semantic_threshold=0.95

    def load_and_label_results(self)->pd.DataFrame:
        all_results=[]
        for results_dir in self.results_dirs:
            results_path=Path(results_dir)/"raw_results.csv"
            if not results_path.exists():
                logger.warning(f"Results file not found: {results_path}");continue
            df=pd.read_csv(results_path)
            rdl=results_dir.lower()
            if 'quantization' in rdl:
                df['compression_method']='quantization'
                if 'quantization_bits' in df.columns:
                    df['compression_details']=df.apply(lambda x:f"b{x['quantization_bits']}_s{int(x.get('sparsity',0)*100)}",axis=1)
            elif 'advanced_pruning' in rdl:
                df['compression_method']='advanced_pruning'
                if 'pruning_method' in df.columns:df['compression_details']=df['pruning_method']
            elif 'pruning' in rdl:
                df['compression_method']='basic_pruning';df['compression_details']='magnitude'
            else:
                df['compression_method']='fine_tuning';df['compression_details']='baseline'
            if 'eval_domain' not in df.columns:df['eval_domain']='id'
            if 'storage_cost_lambda' in df.columns and 'storage_cost_bytes' not in df.columns:df['storage_cost_bytes']=df['storage_cost_lambda']
            if 'is_semantically_equivalent' not in df.columns and 'semantic_similarity' in df.columns:df['is_semantically_equivalent']=df['semantic_similarity']>=self.semantic_threshold
            if 'effective_param_bytes' not in df.columns:
                if 'nonzero_params' in df.columns:
                    if 'storage_cost_bytes' in df.columns:
                        nzm=df['nonzero_params'].replace(0,np.nan).mean()
                        sbm=df['storage_cost_bytes'].mean()
                        bpp=sbm/nzm if nzm and not np.isnan(nzm) else np.nan
                        df['effective_param_bytes']=df['nonzero_params']*bpp
                    else:
                        df['effective_param_bytes']=np.nan
                else:
                    df['effective_param_bytes']=df.get('storage_cost_bytes',pd.Series(np.nan,index=df.index))
            all_results.append(df)
        combined=pd.concat(all_results,ignore_index=True)
        if combined['effective_param_bytes'].isna().any():
            if 'storage_cost_bytes' in combined.columns:
                combined['effective_param_bytes']=combined['effective_param_bytes'].fillna(combined['storage_cost_bytes'])
        def _extract_arch(n:str)->str:
            nl=n.lower()
            if "gpt2-medium" in nl:return "GPT2-Medium"
            if "gpt2" in nl:return "GPT2-Small"
            if "cerebras-gpt-111m" in nl:return "Cerebras-111M"
            if "cerebras-gpt-256m" in nl:return "Cerebras-256M"
            if "cerebras" in nl:return "Cerebras"
            return n.split("_")[0]
        combined['base_architecture']=combined['model_name'].apply(_extract_arch)
        self.combined_df=combined
        return combined

    def _den_bytes(self,row):
        return row['effective_param_bytes'] if not pd.isna(row.get('effective_param_bytes',np.nan)) else row.get('storage_cost_bytes',np.nan)

    def analyze_compression_method(self,method:str,method_df:pd.DataFrame):
        print("\n"+"="*80);print(f"--- DETAILED ANALYSIS FOR {method.upper()} ---");print("="*80)
        theta_values=sorted(method_df['prompt_len_theta'].unique());theta_max=max(theta_values)
        print("\n--- ANALYSIS 1: CAPACITY/DEGRADATION (ID vs OOD at θmax) ---")
        df_max=method_df[method_df['prompt_len_theta']==theta_max]
        if 'compression_details' in df_max.columns:groupby_cols=['model_name','compression_details']
        else:groupby_cols=['model_name']
        summary_rows=[]
        for g,grp in df_max.groupby(groupby_cols):
            if isinstance(g,tuple):mn=g[0];cd=g[1] if len(g)>1 else ''
            else:mn=g;cd=''
            id_g=grp[grp['eval_domain']=='id'] if 'eval_domain' in grp.columns else grp
            ood_g=grp[grp['eval_domain']=='ood'] if 'eval_domain' in grp.columns else pd.DataFrame(columns=grp.columns)
            sr_id=id_g['is_semantically_equivalent'].mean() if len(id_g)>0 else np.nan
            sr_ood=ood_g['is_semantically_equivalent'].mean() if len(ood_g)>0 else np.nan
            deg=np.nan
            if not np.isnan(sr_id) and not np.isnan(sr_ood) and sr_id>0:deg=1-(sr_ood/sr_id)
            den=self._den_bytes(grp.iloc[0])
            nz=grp.get('nonzero_params',pd.Series([np.nan])).iloc[0]
            ef_id=id_g['semantic_similarity'].mean() if len(id_g)>0 else np.nan
            ef_ood=ood_g['semantic_similarity'].mean() if len(ood_g)>0 else np.nan
            ef=ef_ood if not np.isnan(ef_ood) else ef_id
            lat_id=id_g['retrieval_cost_ms'].mean() if len(id_g)>0 else np.nan
            lat_ood=ood_g['retrieval_cost_ms'].mean() if len(ood_g)>0 else np.nan
            lat=lat_ood if not np.isnan(lat_ood) else lat_id
            summary_rows.append({
                'Model (λ)':f"{mn} ({cd})"if cd else mn,
                'Storage (GB)':grp['storage_cost_bytes'].iloc[0]/1e9 if 'storage_cost_bytes' in grp.columns else np.nan,
                'Effective Bytes (GB)':den/1e9 if pd.notna(den) else np.nan,
                'Non-Zero Params (M)':nz/1e6 if pd.notna(nz) else np.nan,
                'Success ID':sr_id,
                'Success OOD':sr_ood,
                'Degradation Rate':deg,
                'Expected Fidelity ID':ef_id,
                'Expected Fidelity OOD':ef_ood,
                'Expected Fidelity':ef,
                'Avg Latency (ms)':lat,
            })
        summary_df=pd.DataFrame(summary_rows)
        print(summary_df.to_string(index=False,float_format="%.4f"))
        print(f"\n--- ANALYSIS 2: RETRIEVAL DEGRADATION (ID vs OOD) ---")
        if 'eval_domain' in method_df.columns:
            rid=method_df[method_df['eval_domain']=='id'].pivot_table(index='model_name',columns='prompt_len_theta',values='is_semantically_equivalent',aggfunc='mean')
            rood=method_df[method_df['eval_domain']=='ood'].pivot_table(index='model_name',columns='prompt_len_theta',values='is_semantically_equivalent',aggfunc='mean')
        else:
            rid=method_df.pivot_table(index='model_name',columns='prompt_len_theta',values='is_semantically_equivalent',aggfunc='mean');rood=pd.DataFrame()
        print("\nID:");print(rid.to_string(float_format="%.4f"))
        if not rood.empty:
            print("\nOOD:");print(rood.to_string(float_format="%.4f"))
        print(f"\n--- ANALYSIS 3: EFFECTIVE INFORMATION RATIO (OOD) ---")
        if 'eval_domain' in method_df.columns:
            text_sents=method_df[method_df['eval_domain']=='ood']['original_sentence'].unique()
        else:
            text_sents=method_df['original_sentence'].unique()
        all_text="\n".join(text_sents)
        bzip2_size=len(bz2.compress(all_text.encode('utf-8'))) if len(all_text)>0 else 1
        scale=1_000_000
        den_series=[]
        for _,r in summary_df.iterrows():
            eff_gb=r['Effective Bytes (GB)'];den_series.append(eff_gb*1e9 if pd.notna(eff_gb) else np.nan)
        den_series=pd.Series(den_series)
        fid_col='Expected Fidelity OOD' if 'Expected Fidelity OOD' in summary_df.columns else 'Expected Fidelity'
        eir_vals=(bzip2_size*summary_df[fid_col])/den_series*scale
        eird=summary_df.copy();eird['EIR_scaled']=eir_vals
        print(f"\nEIR (x{scale:,}) Results:")
        print(eird[['Model (λ)',fid_col,'EIR_scaled']].to_string(index=False,float_format="%.4f"))
        if method=='quantization':self._analyze_quantization_specific(method_df)
        elif 'pruning' in method:self._analyze_pruning_specific(method_df)
        return summary_df,rid,rood,eird

    def _analyze_quantization_specific(self,df):
        print("\n--- QUANTIZATION-SPECIFIC ANALYSIS ---")
        if 'quantization_bits' in df.columns and 'sparsity' in df.columns:
            q=df.groupby(['quantization_bits','sparsity','eval_domain']).agg({'semantic_similarity':'mean','retrieval_cost_ms':'mean','effective_param_bytes':'mean'}).round(4)
            print("\nPerformance by Quantization Configuration:")
            print(q.to_string())

    def _analyze_pruning_specific(self,df):
        print("\n--- PRUNING-SPECIFIC ANALYSIS ---")
        if 'sparsity' in df.columns:
            p=df.groupby(['sparsity','eval_domain']).agg({'semantic_similarity':'mean','retrieval_cost_ms':'mean','nonzero_params':'mean'}).round(4)
            print("\nPerformance by Pruning Amount:")
            print(p.to_string())

    def _grouped_summaries(self):
        gsum={}
        df=self.combined_df
        theta_max=df['prompt_len_theta'].max()
        for (arch,method),g in df.groupby(['base_architecture','compression_method']):
            gm=g[g['prompt_len_theta']==theta_max]
            gid=gm[gm['eval_domain']=='id'] if 'eval_domain' in gm.columns else gm
            good=gm[gm['eval_domain']=='ood'] if 'eval_domain' in gm.columns else pd.DataFrame(columns=gm.columns)
            sidmax=gid.groupby('model_name')['is_semantically_equivalent'].mean().max() if len(gid)>0 else np.nan
            soomax=good.groupby('model_name')['is_semantically_equivalent'].mean().max() if len(good)>0 else np.nan
            rows=[]
            for mn,mg in gm.groupby('model_name'):
                cap=mg['nonzero_params'].iloc[0] if 'nonzero_params' in mg.columns and not pd.isna(mg['nonzero_params'].iloc[0]) else mg['effective_param_bytes'].iloc[0]
                cid=mg[mg['eval_domain']=='id']['is_semantically_equivalent'].mean() if len(mg[mg['eval_domain']=='id'])>0 else np.nan
                cood=mg[mg['eval_domain']=='ood']['is_semantically_equivalent'].mean() if len(mg[mg['eval_domain']=='ood'])>0 else np.nan
                deg_id=np.nan
                if not np.isnan(cid) and not np.isnan(sidmax) and sidmax>0:deg_id=1-(cid/sidmax)
                deg_ood=np.nan
                if not np.isnan(cood) and not np.isnan(soomax) and soomax>0:deg_ood=1-(cood/soomax)
                rows.append((mn,cap,cid,cood,deg_id,deg_ood))
            sdf=pd.DataFrame(rows,columns=['model_name','capacity','success_id','success_ood','deg_id','deg_ood']).sort_values('capacity')
            if len(sdf)>1:
                diffs=sdf['success_ood'].diff().fillna(0)
                if (diffs<-0.02).any():logger.warning(f"non-monotonic OOD scaling in {arch}/{method}")
            gsum[(arch,method)]=sdf
        return gsum

    def create_comprehensive_visualizations(self,method_summaries:Dict):
        print("\n--- CREATING COMPREHENSIVE VISUALIZATIONS ---")
        fig,((ax1,ax2),(ax3,ax4))=plt.subplots(2,2,figsize=(16,12))
        for method,(summary_df,rid,rood,eird) in method_summaries.items():
            color=METHOD_COLORS.get(method,'#000000')
            xvals=summary_df['Effective Bytes (GB)'] if 'Effective Bytes (GB)' in summary_df.columns else summary_df['Storage (GB)']
            yvals=summary_df['Success OOD'] if 'Success OOD' in summary_df.columns else summary_df['Success ID']
            ax1.scatter(xvals,yvals,label=method,color=color,s=100,alpha=0.7,edgecolors='black')
        ax1.set_xlabel('Effective Capacity (GB)');ax1.set_ylabel('OOD Success Rate');ax1.set_title('OOD Performance vs Capacity');ax1.legend();ax1.grid(True,alpha=0.3)
        lat_rows=[]
        for method,(summary_df,_,_,_) in method_summaries.items():
            for _,row in summary_df.iterrows():
                lat_rows.append({'Method':method,'Model':row['Model (λ)'],'Latency':row['Avg Latency (ms)']})
        lat_df=pd.DataFrame(lat_rows)
        sns.boxplot(data=lat_df,x='Method',y='Latency',ax=ax2)
        ax2.set_ylabel('Retrieval Latency (ms)');ax2.set_title('Latency Distribution');ax2.tick_params(axis='x',rotation=45)
        for method,(_,_,_,eird) in method_summaries.items():
            color=METHOD_COLORS.get(method,'#000000')
            ax3.bar(range(len(eird)),eird['EIR_scaled'],label=method,color=color,alpha=0.7)
        ax3.set_ylabel('EIR (×1,000,000)');ax3.set_title('OOD EIR Comparison');ax3.legend();ax3.grid(True,alpha=0.3,axis='y')
        for method,(_,rid,rood,_) in method_summaries.items():
            color=METHOD_COLORS.get(method,'#000000')
            if not rood.empty:
                mean_deg=rood.mean(axis=0);ax4.plot(mean_deg.index,mean_deg.values,label=f"{method} OOD",color=color,marker='o',linewidth=2)
            else:
                mean_deg=rid.mean(axis=0);ax4.plot(mean_deg.index,mean_deg.values,label=f"{method} ID",color=color,marker='o',linewidth=2)
        ax4.set_xlabel('Retrieval Budget θ');ax4.set_ylabel('Avg Success Rate');ax4.set_title('Retrieval Degradation');ax4.legend();ax4.grid(True,alpha=0.3)
        plt.tight_layout();plt.savefig(self.output_dir/'comprehensive_comparison.png',dpi=300,bbox_inches='tight');plt.close()
        self._create_performance_heatmap()

    def _create_performance_heatmap(self):
        df=self.combined_df
        if 'eval_domain' in df.columns:
            df=df[df['eval_domain']=='ood']
        heat=df.pivot_table(index='model_name',columns='compression_method',values='semantic_similarity',aggfunc='mean')
        plt.figure(figsize=(12,10))
        sns.heatmap(heat,annot=True,fmt='.3f',cmap='RdYlGn',cbar_kws={'label':'Avg Semantic Similarity'},vmin=0,vmax=1)
        plt.title('OOD Performance Heatmap: Models × Methods');plt.xlabel('Compression Method');plt.ylabel('Model')
        plt.tight_layout();plt.savefig(self.output_dir/'performance_heatmap.png',dpi=300,bbox_inches='tight');plt.close()

    def fit_scaling_laws(self,method_summaries:Dict):
        print("\n--- SCALING LAW ANALYSIS ACROSS METHODS (OOD) ---")
        def law(N,Qmax,a,g):return Qmax*(1-a*np.power(N,-g))
        fig,ax=plt.subplots(figsize=(12,8))
        for method,(summary_df,_,_,_) in method_summaries.items():
            if len(summary_df)<3:
                print(f"Insufficient data for scaling law fit in {method}");continue
            if 'Effective Bytes (GB)' in summary_df.columns:N=summary_df['Effective Bytes (GB)'].values*1e9
            else:N=summary_df['Storage (GB)'].values*1e9
            if 'Expected Fidelity OOD' in summary_df.columns:Q=summary_df['Expected Fidelity OOD'].values
            else:Q=summary_df['Expected Fidelity'].values
            try:
                popt,_=curve_fit(law,N,Q,p0=[1.0,1e9,0.5],maxfev=8000)
                print(f"\n{method.upper()} Scaling Law: g={popt[2]:.4f}, Qmax={popt[0]:.4f}")
                color=METHOD_COLORS.get(method,'#000000')
                ax.scatter(N,Q,label=f'{method} data',color=color,s=100,alpha=0.7)
                Nf=np.linspace(min(N),max(N),100)
                ax.plot(Nf,law(Nf,*popt),label=f'{method} fit',color=color,linewidth=2,linestyle='--')
            except Exception as e:
                print(f"Could not fit scaling law for {method}: {e}")
        ax.set_xlabel('Effective Capacity N (Bytes)');ax.set_ylabel('OOD Fidelity Q');ax.set_title('Scaling Laws Across Methods');ax.legend();ax.grid(True,alpha=0.3)
        plt.tight_layout();plt.savefig(self.output_dir/'scaling_laws_comparison.png',dpi=300,bbox_inches='tight');plt.close()

    def generate_summary_report(self,method_summaries:Dict):
        rp=self.output_dir/'compression_analysis_report.txt'
        with open(rp,'w') as f:
            f.write("COMPREHENSIVE COMPRESSION METHOD ANALYSIS REPORT\n");f.write("="*80+"\n\n")
            f.write("OVERALL OOD STATISTICS\n");f.write("-"*40+"\n")
            df=self.combined_df
            if 'eval_domain' in df.columns:df=df[df['eval_domain']=='ood']
            overall=df.groupby('compression_method').agg({'semantic_similarity':['mean','std','max'],'retrieval_cost_ms':['mean','std','min'],'effective_param_bytes':'mean','is_semantically_equivalent':'mean'}).round(4)
            f.write(overall.to_string()+"\n\n")
            f.write("BEST CONFIGURATIONS (OOD)\n");f.write("-"*40+"\n")
            if len(df)>0:
                bs=df.loc[df['semantic_similarity'].idxmax()]
                f.write(f"Best OOD Similarity: {bs['model_name']} ({bs['compression_method']}) - {bs['semantic_similarity']:.4f}\n")
                sp=df.loc[df['retrieval_cost_ms'].idxmin()]
                f.write(f"Fastest Retrieval: {sp['model_name']} ({sp['compression_method']}) - {sp['retrieval_cost_ms']:.2f}ms\n")
                be_idx=(df['semantic_similarity']/df['effective_param_bytes']).idxmax()
                be=df.loc[be_idx]
                f.write(f"Most Efficient: {be['model_name']} ({be['compression_method']}) - {be['semantic_similarity']/be['effective_param_bytes']:.6e} sim/byte\n")
            f.write("\n"+"="*80+"\n");f.write("Analysis complete.\n")
        print(f"\nSummary report saved to {rp}")

    def run_full_analysis(self):
        print("COMPREHENSIVE COMPRESSION METHOD ANALYSIS");print("="*80)
        print("\nLoading results from all experiments...")
        self.load_and_label_results()
        self.combined_df.to_csv(self.output_dir/'combined_results.csv',index=False)
        print(f"Combined results saved to {self.output_dir/'combined_results.csv'}")
        method_summaries={}
        for method in self.combined_df['compression_method'].unique():
            mdf=self.combined_df[self.combined_df['compression_method']==method]
            summaries=self.analyze_compression_method(method,mdf)
            method_summaries[method]=summaries
        self.create_comprehensive_visualizations(method_summaries)
        gs=self._grouped_summaries()
        for (arch,method),dfg in gs.items():
            dfg.to_csv(self.output_dir/f"group_{arch}_{method}.csv",index=False)
        self.fit_scaling_laws(method_summaries)
        self.generate_summary_report(method_summaries)
        print("\n"+"="*80);print("ANALYSIS COMPLETE!");print(f"All results saved to: {self.output_dir}");print("="*80)

def main():
    p=argparse.ArgumentParser(description="Comprehensive analysis of LLM compression methods")
    p.add_argument('--results-dirs',nargs='+',required=True,help='Directories containing experiment results')
    p.add_argument('--output-dir',type=str,default='results/comprehensive_analysis',help='Output directory')
    p.add_argument('--semantic-threshold',type=float,default=0.95,help='ID semantic threshold (fallback)')
    a=p.parse_args()
    an=CompressionAnalyzer(a.results_dirs,a.output_dir)
    an.semantic_threshold=a.semantic_threshold
    an.run_full_analysis()

if __name__=="__main__":
    main()

