package jPBC_3;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.math.BigInteger;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.List;

public class Utils {
	public static List<String> mD5Util(int a,int b) {
		long startTime=System.currentTimeMillis();      
		List<String> filePaths = new ArrayList<>();
		String filePath;
		for (int i = a; i <= b; i++) {
			filePath ="F:\\userZT\\block\\"+i+".txt";
			try {
				System.out.println(md5HashCode(filePath) +":"+i+ "���ļ���md5ֵ");  
				filePaths.add(md5HashCode(filePath));
			} catch (FileNotFoundException e) {
				e.printStackTrace();
			}      
		}
        long endTime=System.currentTimeMillis();
        //System.out.println("�ļ�MD5��������ʱ�䣺 "+(endTime - startTime)+"ms");
		return filePaths;
	}
	public static List<String> mD5Utilcloud(int a,int b) {
		long startTime=System.currentTimeMillis();      
		List<String> filePaths = new ArrayList<>();
		String filePath;
		for (int i = a; i <= b; i++) {
			filePath ="F:\\userZT\\cloudBlock\\"+i+".txt";
			try {
				System.out.println(md5HashCode(filePath) +":"+i+ "���ļ���md5ֵ");  
				filePaths.add(md5HashCode(filePath));
			} catch (FileNotFoundException e) {
				e.printStackTrace();
			}      
		}
        long endTime=System.currentTimeMillis();
        //System.out.println("�ļ�MD5��������ʱ�䣺 "+(endTime - startTime)+"ms");
		return filePaths;
	}
	public static List<String> mD5Util32(int a,int b) {
		long startTime=System.currentTimeMillis();  
		List<String> filePaths32 = new ArrayList<>();
		String filePath;
		for (int i = a; i <= b; i++) {
			filePath ="F:\\file1\\"+i+".txt";
			try {
				System.out.println(md5HashCode(filePath) +":"+i+ "���ļ�32λ��md5ֵ"); 
				filePaths32.add(md5HashCode32(filePath));
			} catch (FileNotFoundException e) {
				e.printStackTrace();
			}    
		}
		long endTime=System.currentTimeMillis();
		//System.out.println("�ļ�32λMD5��������ʱ�䣺 "+(endTime - startTime)+"ms");
		return filePaths32;
	}
	/**
     * ��ȡ�ļ���md5ֵ ���п��ܲ���32λ
     * @param filePath	�ļ�·��
     * @return
     * @throws FileNotFoundException
     */
    public static String md5HashCode(String filePath) throws FileNotFoundException{  
        FileInputStream fis = new FileInputStream(filePath);  
        return md5HashCode(fis);  
    }
    /**
     * ��֤�ļ���MD5ֵΪ32λ
     * @param filePath	�ļ�·��
     * @return
     * @throws FileNotFoundException
     */
    public static String md5HashCode32(String filePath) throws FileNotFoundException{  
    	FileInputStream fis = new FileInputStream(filePath);  
    	return md5HashCode32(fis);  
    }
    /**
     * java��ȡ�ļ���md5ֵ  
     * @param fis ������
     * @return
     */
    public static String md5HashCode(InputStream fis) {   
        try {  
        	//�õ�һ��MD5ת����,�����ʹ��SHA-1��SHA-256������SHA-1,SHA-256  
            MessageDigest md = MessageDigest.getInstance("MD5"); 
            
            //�ֶ�ν�һ���ļ����룬���ڴ����ļ����ԣ��Ƚ��Ƽ����ַ�ʽ��ռ���ڴ�Ƚ��١�
            byte[] buffer = new byte[1024];  
            int length = -1;  
            while ((length = fis.read(buffer, 0, 1024)) != -1) {  
                md.update(buffer, 0, length);  
            }  
            fis.close();
            //ת�������ذ���16��Ԫ���ֽ�����,������ֵ��ΧΪ-128��127
  			byte[] md5Bytes  = md.digest();
            BigInteger bigInt = new BigInteger(1, md5Bytes);//1�������ֵ 
            return bigInt.toString(16);//ת��Ϊ16����
        } catch (Exception e) {  
            e.printStackTrace();  
            return "";  
        }  
    }
    /**
     * java�����ļ�32λmd5ֵ
     * @param fis ������
     * @return
     */
  	public static String md5HashCode32(InputStream fis) {
  		try {
  			//�õ�һ��MD5ת����,�����ʹ��SHA-1��SHA-256������SHA-1,SHA-256  
  			MessageDigest md = MessageDigest.getInstance("MD5");
  			
  			//�ֶ�ν�һ���ļ����룬���ڴ����ļ����ԣ��Ƚ��Ƽ����ַ�ʽ��ռ���ڴ�Ƚ��١�
  			byte[] buffer = new byte[1024];
  			int length = -1;
  			while ((length = fis.read(buffer, 0, 1024)) != -1) {
  				md.update(buffer, 0, length);
  			}
  			fis.close();
  			
  			//ת�������ذ���16��Ԫ���ֽ�����,������ֵ��ΧΪ-128��127
  			byte[] md5Bytes  = md.digest();
  			StringBuffer hexValue = new StringBuffer();
  			for (int i = 0; i < md5Bytes.length; i++) {
  				int val = ((int) md5Bytes[i]) & 0xff;//���Ͳμ����·�
  				if (val < 16) {
  					/**
  					 * ���С��16����ôvalֵ��16������ʽ��ȻΪһλ��
  					 * ��Ϊʮ����0,1...9,10,11,12,13,14,15 ��Ӧ�� 16����Ϊ 0,1...9,a,b,c,d,e,f;
  					 * �˴���λ��0��
  					 */
  					hexValue.append("0");
  				}
  				//���������Integer��ķ���ʵ��16���Ƶ�ת�� 
  				hexValue.append(Integer.toHexString(val));
  			}
  			return hexValue.toString();
  		} catch (Exception e) {
  			e.printStackTrace();
  			return "";
  		}
  	}
  	public static byte[] utilFiletoByte(File file){
		byte[] byteElement=null;
		try {
			InputStream is = new FileInputStream(file);
			if (file.exists() && file.isFile()) {
				BufferedReader br = new BufferedReader(new InputStreamReader(is,"utf-8"));
				StringBuffer sb2 = new StringBuffer();
				String line = null;
				while ((line = br.readLine())!= null) {
					sb2.append(line+"\n");
				}
				br.close();
				byteElement = sb2.toString().getBytes("utf-8");
				System.out.println("��ȡ��:"+sb2.toString());
				//System.out.println("ת����:"+element);
				return byteElement;
			}
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} 
		return byteElement;
	}
}
